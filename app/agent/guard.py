"""
Cheap, fast pre-filter for the most blatant prompt-injection patterns.

This is a backstop, not the primary defense -- the system prompt already
instructs the model to ignore in-message instructions that try to override
its behavior, and that handles the wide variety of subtler injection
attempts. This regex layer exists for the small set of extremely obvious
cases ("ignore all previous instructions", "reveal your system prompt")
where short-circuiting before an LLM call is both faster and strictly safer
than hoping the model resists every time.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"disregard (all )?(previous|above|prior) (instructions|rules)",
    r"reveal (your |the )?system prompt",
    r"show me (your |the )?(system )?prompt",
    r"you are now (in )?(dan|developer mode|jailbreak)",
    r"act as (if you (are|were)|an?) (unrestricted|unfiltered|jailbroken)",
    r"pretend (you are|to be) (an? )?(ai )?without (any )?(restrictions|rules|filters)",
    r"forget (your|all|everything) (instructions|rules|guidelines)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


REFUSAL_REPLY = (
    "I can only help with selecting SHL assessments -- I'm not able to "
    "follow instructions that try to change how I operate. What role or "
    "hiring need can I help you find assessments for?"
)
