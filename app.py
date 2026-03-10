import os
import uuid
import pickle
from typing import List, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

AUX_MODEL = os.getenv("AUX_MODEL", "llama3.2:1b")

INDEX_NAME = os.getenv("INDEX_NAME", "quran-tafseer-jal")
NAMESPACE = os.getenv("NAMESPACE", "quran_jal")
BM25_PATH = os.getenv("BM25_PATH", "bm25_quran_jal.pkl")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment variables.")


app = FastAPI(title="Quran RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


chat_store: Dict[str, List[Dict[str, str]]] = {}

pc = Pinecone(api_key=PINECONE_API_KEY)
index_jal = pc.Index(INDEX_NAME)

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

with open(BM25_PATH, "rb") as f:
    bm25_jal = pickle.load(f)

retriever_jal = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_jal,
    index=index_jal,
    namespace=NAMESPACE,
    top_k=10,
    alpha=0.7,
)


answer_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.0,
    base_url=OLLAMA_URL,
)

aux_llm = ChatOllama(
    model=AUX_MODEL,
    temperature=0.0,
    base_url=OLLAMA_URL,
)


sys_prompt_first = """
You are a retrieval-grounded Quran QA assistant (English).

You MUST answer using ONLY the provided retrieved context.
Treat the retrieved context as the ONLY source of truth.
The retrieved context may include Qur'an translation text and Tafsir (e.g., Tafsir al-Jalalayn).

Rules (strict):
1) Use ONLY the retrieved context. Do NOT use outside knowledge, memory, or guesses.
2) Do NOT invent or guess: verse numbers, surah names, citations, quotes, narrations, or any details not explicitly present in the retrieved context.
3) If you cannot answer the question with explicit support from the retrieved context, you MUST output exactly:
Answer: INSUFFICIENT_CONTEXT
Evidence: NONE
Queries:
- <query 1>
- <query 2>
- <query 3>
Then stop.
4) Every factual claim in your Answer MUST be supported by Evidence.
   - Evidence MUST be copied verbatim from the retrieved context.
   - Do NOT paraphrase evidence.
   - If you cannot provide a verbatim quote that supports the claim, output INSUFFICIENT_CONTEXT.
5) Distinguish sources clearly:
   - If the supporting quote is Qur'an translation text, label it as [Quran].
   - If the supporting quote is tafsir, label it as [Tafsir] and treat it as explanation, not a direct Qur'an quote.
6) Citations MUST be taken from the retrieved context metadata.
   - Only cite a surah/ayah range if it is explicitly present in the retrieved context.
   - If the context does not include surah/ayah metadata, do NOT guess it; instead output INSUFFICIENT_CONTEXT.
7) If retrieved passages are ambiguous or contradictory, the Answer MUST start with:
AMBIGUOUS:
and you must provide evidence quotes for each interpretation/view.
8) Keep the tone respectful and neutral.
   - Do not issue fatwas or absolute legal rulings.
   - If the question requires scholarly/legal judgment beyond the retrieved context, output INSUFFICIENT_CONTEXT.

Output format (STRICT: no extra lines, no extra commentary):
Answer: <1-3 short sentences max, unless INSUFFICIENT_CONTEXT>
Evidence:
- [Quran] "(Surah X Ayah A-B)" "<verbatim quote from retrieved context>"
- [Tafsir] "(Surah X Ayah A-B)" "<verbatim quote from retrieved context>"
- ...
"""

sys_prompt_followup = """
You are a retrieval-grounded Quran QA assistant (English).

You will receive:
1) a conversation summary
2) a new user question
3) retrieved context

The conversation summary is ONLY for resolving the user's follow-up intent.
It is NOT evidence.
If a fact appears in the conversation summary but is not explicitly supported by the retrieved context, treat it as unverified and do not use it in the answer.

The retrieved context is the ONLY source of truth.

Rules (strict):
1) Use ONLY the retrieved context for factual claims.
2) Do NOT use outside knowledge, memory, or guesses.
3) Do NOT invent or guess: verse numbers, surah names, citations, quotes, narrations, or any missing details not explicitly present in the retrieved context.
4) If the new question cannot be answered with explicit support from the retrieved context, you MUST output exactly:
Answer: INSUFFICIENT_CONTEXT
Evidence: NONE
Queries:
- <query 1>
- <query 2>
- <query 3>
Then stop.
5) Every factual claim in your Answer MUST be supported by Evidence.
   - Evidence MUST be copied verbatim from the retrieved context.
   - Do NOT paraphrase evidence.
   - If you cannot provide a verbatim quote that supports the claim, output INSUFFICIENT_CONTEXT.
6) Distinguish sources clearly:
   - If the supporting quote is Qur'an translation text, label it as [Quran].
   - If the supporting quote is tafsir, label it as [Tafsir] and treat it as explanation, not a direct Qur'an quote.
7) Citations MUST be taken from the retrieved context metadata.
   - Only cite a surah/ayah range if it is explicitly present in the retrieved context.
   - If the context does not include surah/ayah metadata, do NOT guess it; instead output INSUFFICIENT_CONTEXT.
8) If retrieved passages are ambiguous or contradictory, the Answer MUST start with:
AMBIGUOUS:
and you must provide evidence quotes for each interpretation/view.
9) Keep the tone respectful and neutral.
   - Do not issue fatwas or absolute legal rulings.
   - If the question requires scholarly/legal judgment beyond the retrieved context, output INSUFFICIENT_CONTEXT.

Output format (STRICT: no extra commentary):
Answer: <1-3 short sentences max, unless INSUFFICIENT_CONTEXT>
Evidence:
- [Quran] "(Surah X Ayah A-B)" "<verbatim quote from retrieved context>"
- [Tafsir] "(Surah X Ayah A-B)" "<verbatim quote from retrieved context>"
- ...
"""

summary_sys_prompt = """
You summarize prior chat turns for a retrieval-grounded Quran QA system.

Goals:
- preserve the user's main topic and intent
- preserve named entities, prophets, surahs, ayahs, and concepts if mentioned
- preserve unresolved references needed for follow-up questions
- preserve only useful context for the next turn
- do not add outside knowledge
- do not add facts not present in the chat history
- be concise

Output:
Return only a short plain-text summary.
"""

rag_query_sys_prompt = """
You create a retrieval query for a Quran + Tafsir RAG system.

You will receive:
1) a summary of prior conversation
2) a new user question

Your task:
Generate one concise search query that combines the important prior context with the user's new question.

Rules:
- output only the query
- do not explain
- do not answer the question
- include important names, topics, and relationships when useful
- keep it concise but specific
"""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: int = 10
    alpha: float = 0.7


class SourceDoc(BaseModel):
    surah_no: Optional[int] = None
    ayah_start: Optional[int] = None
    ayah_end: Optional[int] = None
    text: str


class AskResponse(BaseModel):
    session_id: str
    question: str
    generated_rag_query: Optional[str] = None
    history_summary: Optional[str] = None
    answer: str
    sources: List[SourceDoc]


def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in chat_store:
        return session_id

    new_session_id = session_id or str(uuid.uuid4())
    if new_session_id not in chat_store:
        chat_store[new_session_id] = []
    return new_session_id


def format_docs(docs) -> str:
    parts = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        s = md.get("surah_no", "?")
        a0 = md.get("ayah_start", "?")
        a1 = md.get("ayah_end", "?")
        src = md.get("source_type", "")
        label = f"[{src}] " if src else ""
        parts.append(f'{label}(Surah {s} Ayah {a0}-{a1})\n{d.page_content}')
    return "\n\n".join(parts)


def docs_to_sources(docs) -> List[SourceDoc]:
    out = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        out.append(
            SourceDoc(
                surah_no=md.get("surah_no"),
                ayah_start=md.get("ayah_start"),
                ayah_end=md.get("ayah_end"),
                text=d.page_content,
            )
        )
    return out


def history_to_text(history: List[Dict[str, str]]) -> str:
    lines = []
    for turn in history:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""

    history_text = history_to_text(history)

    messages = [
        ("system", summary_sys_prompt),
        (
            "user",
            f"Summarize this previous conversation for future retrieval-grounded QA:\n\n{history_text}",
        ),
    ]
    resp = aux_llm.invoke(messages)
    return resp.content.strip()


def build_rag_query(summary: str, new_question: str) -> str:
    if not summary.strip():
        return new_question.strip()

    messages = [
        ("system", rag_query_sys_prompt),
        (
            "user",
            f"Conversation summary:\n{summary}\n\nNew question:\n{new_question}",
        ),
    ]
    resp = aux_llm.invoke(messages)
    query = resp.content.strip()

    if not query:
        return new_question.strip()
    return query


def retrieve_docs(query: str, top_k: int = 10, alpha: float = 0.7):
    if hasattr(retriever_jal, "top_k"):
        retriever_jal.top_k = top_k
    if hasattr(retriever_jal, "alpha"):
        retriever_jal.alpha = alpha
    return retriever_jal.invoke(query)


def answer_first_turn(question: str, docs) -> str:
    context = format_docs(docs)

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer using ONLY the retrieved context."
    )

    resp = answer_llm.invoke(
        [
            ("system", sys_prompt_first),
            ("user", user_prompt),
        ]
    )
    return resp.content.strip()


def answer_followup(summary: str, question: str, docs) -> str:
    context = format_docs(docs)

    user_prompt = (
        f"Conversation summary:\n{summary}\n\n"
        f"New question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer the new question using ONLY the retrieved context.\n"
        "Use the conversation summary only to understand what the user is referring to.\n"
        "Do NOT use the summary as evidence."
    )

    resp = answer_llm.invoke(
        [
            ("system", sys_prompt_followup),
            ("user", user_prompt),
        ]
    )
    return resp.content.strip()



@app.get("/")
def root():
    return {"message": "Quran RAG Chat API is running"}


@app.post("/ask", response_model=AskResponse)
@app.post("/chat", response_model=AskResponse)
def chat(req: AskRequest):
    session_id = get_or_create_session(req.session_id)
    history = chat_store[session_id]

    try:
        if len(history) == 0:
            rag_query = req.question.strip()
            docs = retrieve_docs(rag_query, top_k=req.top_k, alpha=req.alpha)
            answer = answer_first_turn(req.question, docs)
            summary = None

        else:
            summary = summarize_history(history)
            rag_query = build_rag_query(summary, req.question)
            docs = retrieve_docs(rag_query, top_k=req.top_k, alpha=req.alpha)
            answer = answer_followup(summary, req.question, docs)

        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": answer})

        return AskResponse(
            session_id=session_id,
            question=req.question,
            generated_rag_query=rag_query,
            history_summary=summary,
            answer=answer,
            sources=docs_to_sources(docs),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in chat_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": chat_store[session_id]}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in chat_store:
        del chat_store[session_id]
    return {"message": "Session cleared", "session_id": session_id}