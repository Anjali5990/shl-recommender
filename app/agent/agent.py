"""
Agent orchestration -- the piece that ties retrieval, the LLM, and output
validation together for one /chat call.

Per-request flow:
1. Fast injection backstop (guard.py) -- short-circuits obvious attacks
   without spending an LLM call.
2. Reconstruct conversation state from the stateless message history:
   - query_text: all user turns concatenated, for retrieval
   - current_shortlist: catalog items the assistant has already named in
     past replies (re-derived by scanning its own past replies against the
     real catalog) -- this is the ONLY thing that survives between calls,
     since the API's message schema has no field for `recommendations`.
     prompts.py now instructs the LLM to always name every shortlist item
     by exact name in its reply text, specifically so this reconstruction
     doesn't silently lose items (see v2 note in prompts.py).
   - pool: additional retrieval candidates NOT already on the shortlist,
     kept as a separate labeled section from current_shortlist so the LLM
     isn't visually tempted to re-recommend on pure COMPARE turns.
3. Retrieve candidates (retriever.py) = semantic/lexical top-K ∪ anything
   named in the latest user turn, minus whatever's already in
   current_shortlist. Shortlist ∪ pool is the ONLY set of assessments the
   LLM is allowed to choose from.
4. Call the LLM with shortlist + pool + full conversation text.
5. Validate the LLM's JSON: every recommended item must resolve to a real
   catalog record (via retriever.find_by_name/find_by_url); anything that
   doesn't is dropped, never passed through. This is what makes a
   hallucinated name/URL structurally impossible to return to the user,
   regardless of what the model outputs.
6. One repair retry on malformed JSON; a safe, schema-compliant fallback
   reply if the LLM is unreachable or still fails after retry.
"""
from app.agent import prompts
from app.agent.guard import looks_like_injection, REFUSAL_REPLY
from app.agent.llm_client import complete_json, LLMError
from app.models import ChatResponse, Message, Recommendation
from app.retrieval.retriever import get_retriever

MAX_CANDIDATES_SENT = 8
TOP_K_RETRIEVED = 6


def _history_text(messages: list) -> str:
    lines = []
    for m in messages:
        role = "User" if m.role == "user" else "Agent"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _build_shortlist_and_pool(messages: list, retriever):
    """Returns (current_shortlist, pool) as two disjoint lists.

    current_shortlist: every catalog item whose exact name has appeared in
    ANY past assistant reply, de-duplicated, in first-seen order. This is
    the sole surviving record of "what's already been recommended" -- see
    the statelessness note in prompts.py.

    pool: fresh retrieval candidates (semantic/lexical top-K + anything
    named in the latest user turn) that are NOT already in the shortlist,
    truncated to fit the remaining candidate budget.
    """
    user_turns = [m.content for m in messages if m.role == "user"]
    assistant_turns = [m.content for m in messages if m.role == "assistant"]
    latest_user_message = user_turns[-1] if user_turns else ""
    query_text = "\n".join(user_turns)

    current_shortlist = []
    shortlist_ids = set()
    for reply in assistant_turns:
        for item in retriever.extract_mentioned_items(reply):
            if item["id"] not in shortlist_ids:
                shortlist_ids.add(item["id"])
                current_shortlist.append(item)

    retrieved = retriever.retrieve(
        query_text=query_text or latest_user_message,
        latest_user_message=latest_user_message,
        top_k=TOP_K_RETRIEVED,
    )

    pool = []
    pool_ids = set()
    for item in retrieved:
        if item["id"] not in shortlist_ids and item["id"] not in pool_ids:
            pool_ids.add(item["id"])
            pool.append(item)

    remaining_budget = max(MAX_CANDIDATES_SENT - len(current_shortlist), 0)
    return current_shortlist, pool[:remaining_budget]


def _validate_and_ground(raw: dict, retriever) -> ChatResponse:
    reply = str(raw.get("reply", "")).strip()
    if not reply:
        reply = "Could you tell me a bit more about the role you're hiring for?"

    grounded = []
    for rec in raw.get("recommendations", []) or []:
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        item = retriever.find_by_url(url) or retriever.find_by_name(name)
        if item is None:
            continue  # hallucinated / unmatched -- dropped, never shown to user
        grounded.append(
            Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=",".join(item["test_type"]) or "unspecified",
            )
        )
        if len(grounded) >= 10:
            break

    end_of_conversation = bool(raw.get("end_of_conversation", False))

    return ChatResponse(
        reply=reply,
        recommendations=grounded,
        end_of_conversation=end_of_conversation,
    )


def _fallback_response(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def run_chat(messages: list[Message]) -> ChatResponse:
    if not messages or messages[-1].role != "user":
        return _fallback_response(
            "I didn't receive a question to respond to -- what role or "
            "hiring need can I help you find assessments for?"
        )

    latest = messages[-1].content

    if looks_like_injection(latest):
        return _fallback_response(REFUSAL_REPLY)

    retriever = get_retriever()
    current_shortlist, pool = _build_shortlist_and_pool(messages, retriever)

    user_prompt = prompts.build_user_turn(
        _history_text(messages), current_shortlist, pool
    )

    try:
        raw = complete_json(prompts.SYSTEM_PROMPT, user_prompt)
    except LLMError:
        try:
            raw = complete_json(
                prompts.SYSTEM_PROMPT,
                user_prompt + "\n\nReminder: output ONLY the JSON object, nothing else.",
            )
        except LLMError:
            pass
            return _fallback_response(
                "I'm having trouble reaching the recommendation engine right "
                "now. Could you try again in a moment?"
            )

    return _validate_and_ground(raw, retriever)
