import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.agent.agent import run_chat
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval.retriever import get_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_up():
    # Load the catalog + fit TF-IDF (and dense embeddings, if built) once at
    # startup, not on the first request -- keeps /chat inside its 30s budget
    # even on a cold instance. /health tolerates up to 2 min for this per
    # the assignment's cold-start allowance.
    logger.info("Warming up retriever...")
    get_retriever()
    logger.info("Retriever ready.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return run_chat(request.messages)
