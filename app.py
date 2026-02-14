import asyncio
import json
import os
import re
import threading
import hashlib
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, render_template, request, jsonify, session

from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


# ============================================================
# Flask setup
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


def get_ui_messages() -> List[dict]:
    # messages format: [{"role":"user|assistant", "content":"...", "question_type": "..."}]
    return session.get("messages", [])


def set_ui_messages(msgs: List[dict]) -> None:
    session["messages"] = msgs


# ============================================================
# Config
# ============================================================
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "machine-learning"

EMBED_MODEL_NAME = "qwen3-embedding:4b"
LLM_MODEL_NAME = "qwen3:1.7b"

TOP_K_DEFAULT = 10
DIFFICULTY_TRIGGER_COUNT = 3
QUIZ_MAX_ATTEMPTS = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_PATH = os.path.join(BASE_DIR, "chat_log.json")
QUIZ_LOG_PATH = os.path.join(BASE_DIR, "quiz_log.json")


# ============================================================
# Utilities
# ============================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_list(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def _write_json_list(path: str, data_list: List[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_chat_log(entry: dict) -> None:
    data = _read_json_list(CHAT_LOG_PATH)
    data.append(entry)
    _write_json_list(CHAT_LOG_PATH, data)


def append_quiz_log(entry: dict) -> None:
    data = _read_json_list(QUIZ_LOG_PATH)
    data.append(entry)
    _write_json_list(QUIZ_LOG_PATH, data)


def get_recent_history(n: int = 12) -> List[dict]:
    data = _read_json_list(CHAT_LOG_PATH)
    return data[-n:]


def get_asked_quiz_questions() -> List[str]:
    data = _read_json_list(QUIZ_LOG_PATH)
    return [x.get("quiz_question", "") for x in data if x.get("quiz_question")]


def _is_affirmative(text: str) -> bool:
    t = text.strip().lower()
    return t in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "teach", "teach you"}


def _is_negative(text: str) -> bool:
    t = text.strip().lower()
    return t in {"no", "n", "nope", "nah", "not now", "later", "cancel", "stop"}


def _looks_like_url(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://")


# ============================================================
# Retrieval (your working code)
# ============================================================
ollama_client = Client(host=OLLAMA_URL)
qdrant = QdrantClient(url=QDRANT_URL)


def embed_query(text: str) -> List[float]:
    resp = ollama_client.embed(model=EMBED_MODEL_NAME, input=text)
    return resp["embeddings"][0]


def retrieve_top_k(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    qvec = embed_query(query)
    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k,
        with_payload=True,
    )
    results = []
    for p in res.points:
        payload = p.payload or {}
        results.append(
            {
                "score": float(p.score),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
            }
        )
    return results


def format_context(passages: List[Dict[str, Any]]) -> str:
    out = []
    for i, p in enumerate(passages, 1):
        out.append(f"[{i}] source={p['source']} score={p['score']:.4f}\n{p['text']}")
    return "\n\n".join(out)


def retrieval_metadata(passages: List[Dict[str, Any]]) -> Tuple[bool, List[str], Optional[float]]:
    texts = [p.get("text", "") for p in passages]
    retrieved_any_text = any(t.strip() for t in texts)

    sources, seen = [], set()
    for p in passages:
        s = p.get("source", "unknown")
        if s not in seen:
            seen.add(s)
            sources.append(s)
        if len(sources) >= 8:
            break

    top_score = max(
        (p.get("score", None) for p in passages if isinstance(p.get("score", None), (int, float))),
        default=None,
    )
    return retrieved_any_text, sources, top_score


# ============================================================
# AutoGen helpers + agents
# ============================================================
def _extract_json(text: str) -> dict:
    """
    Robust JSON extractor that handles:
    - Proper JSON objects
    - JSON object embedded in text
    - Raw JSON list (wrapped into {"facts": [...]})
    """

    text = (text or "").strip()

    # Case 1: whole content is a JSON object
    if text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Case 2: JSON object embedded somewhere in the text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Case 3: whole content is a JSON list -> wrap
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed_list = json.loads(text)
            return {"facts": parsed_list}
        except Exception:
            pass

    raise ValueError(f"Expected JSON but got: {text[:200]}")


async def _agent_json(agent: AssistantAgent, prompt: str) -> dict:
    """
    Run an agent and parse its response as JSON using _extract_json().
    """
    res = await agent.run(task=prompt)

    # Prefer the last message from the agent
    for msg in reversed(res.messages):
        if getattr(msg, "source", None) == agent.name and hasattr(msg, "content"):
            return _extract_json(str(msg.content))

    # Fallback: any message with content
    for msg in reversed(res.messages):
        if hasattr(msg, "content"):
            return _extract_json(str(msg.content))

    raise RuntimeError("No agent output")


# IMPORTANT: One global model client, used from a persistent event loop
model_client = OllamaChatCompletionClient(model=LLM_MODEL_NAME, host=OLLAMA_URL)

difficulty_agent = AssistantAgent(
    name="difficulty_grader",
    model_client=model_client,
    system_message=(
        "You classify machine learning questions.\n\n"
        "EASY:\n"
        "- Definitions\n"
        "- Intro concepts\n"
        "- High-level explanation\n\n"
        "DIFFICULT:\n"
        "- Mathematical reasoning\n"
        "- Theoretical guarantees\n"
        "- Convergence proofs\n"
        "- Generalization theory\n"
        "- Optimization dynamics\n\n"
        "If a question requires advanced math or theory, classify as difficult.\n\n"
        "Return ONLY:\n"
        "{\"difficulty\":\"easy\"}\n"
        "or\n"
        "{\"difficulty\":\"difficult\"}"
    ),
)



answer_agent = AssistantAgent(
    name="rag_answerer",
    model_client=model_client,
    system_message=(
        "You are a strict retrieval-augmented question answering system.\n\n"

        "You will be given CONTEXT and a QUESTION.\n\n"

        "Rules:\n"
        "1) If the CONTEXT clearly contains enough information to answer the QUESTION, "
        "produce a direct, clear, complete answer.\n"
        "2) If the CONTEXT does NOT contain sufficient relevant information, "
        "output exactly: I don't know.\n"
        "3) Do NOT use outside knowledge.\n"
        "4) Do NOT mention the word CONTEXT.\n"
        "5) Do NOT explain your reasoning process.\n"
        "6) Do NOT hedge unnecessarily if the answer is supported.\n\n"

        "Be confident when information is present.\n"
        "Be strict when information is missing."
    ),
)

quiz_gen_agent = AssistantAgent(
    name="quiz_generator",
    model_client=model_client,
    system_message=(
        "Generate ONE quiz question to help the user learn. "
        "Base it on the user's recent chat topics. "
        "It must NOT repeat any question from already_asked_quiz_questions."
        "Return STRICT JSON only with keys: quiz_question, topic, difficulty (easy|medium|hard). "
        "The quiz_question must be a single clear question."
    ),
)

quiz_grade_agent = AssistantAgent(
    name="quiz_grader",
    model_client=model_client,
    system_message=(
        "Grade the user's quiz answer. "
        "Return STRICT JSON only with keys: correct (true|false), rationale, ideal_answer. "
        "Be strict but fair."
    ),
)

# --- Teach mode agents ---
knowledge_extractor_agent = AssistantAgent(
    name="knowledge_extractor",
    model_client=model_client,
    system_message=(
        "You MUST output STRICT VALID JSON.\n"
        "Return ONLY this format:\n"
        "{\n"
        '  "facts": ["fact1", "fact2", "fact3"]\n'
        "}\n"
        "No explanation.\n"
        "No markdown.\n"
        "No text outside JSON.\n"
        "Facts must be short, atomic, factual statements."
    ),
)

knowledge_verifier_agent = AssistantAgent(
    name="knowledge_verifier",
    model_client=model_client,
    system_message=(
        "Verify the factual correctness and internal consistency of the given facts.\n"
        "Return STRICT JSON only with keys:\n"
        "  valid: true/false\n"
        "  cleaned_facts: list of facts that are safe and consistent\n"
        "Rules:\n"
        "- If unsure, mark valid=false.\n"
        "- Remove any speculative or contradictory facts.\n"
    ),
)


# ============================================================
# Teach-mode + quiz-mode state (GLOBAL)
# NOTE: This is shared across users. If you want per-user, move to Flask session.
# ============================================================
@dataclass
class SessionState:
    difficult_count: int = 0

    quiz_active: bool = False
    quiz_question: Optional[str] = None
    quiz_attempts: int = 0

    teach_active: bool = False          # user is in teach flow
    teach_stage: str = "none"           # "awaiting_consent" | "awaiting_content" | "none"
    teach_question: Optional[str] = None  # the question we couldn't answer


state = SessionState()


# ============================================================
# Qdrant upsert for learned facts
# ============================================================
def _fact_id_int(fact: str) -> int:
    # stable-ish int id; avoids duplicates across runs
    h = hashlib.sha1(fact.encode("utf-8")).hexdigest()
    return int(h[:16], 16)  # fits in 64-bit


def add_knowledge_to_qdrant(facts: List[str], topic: str, source_text: str) -> int:
    """
    Adds facts to Qdrant as separate points.
    Returns number of facts upserted.
    """
    points: List[qmodels.PointStruct] = []
    for fact in facts:
        fact_clean = str(fact).strip()
        if not fact_clean:
            continue

        vec = embed_query(fact_clean)
        pid = _fact_id_int(fact_clean)

        payload = {
            "text": fact_clean,
            "source": "user_provided",
            "source_type": "user_provided",
            "topic": topic,
            "added_at": now_utc_iso(),
            "teach_origin_question": topic,
            "teach_raw_input": source_text[:2000],
        }
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))

    if not points:
        return 0

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


# ============================================================
# Core async logic
# ============================================================
async def grade_difficulty(question: str) -> str:
    j = await _agent_json(difficulty_agent, f"User question:\n{question}\nReturn JSON only.")
    d = str(j.get("difficulty", "easy")).strip().lower()
    if "difficult" in d:
        return "difficult"
    return "easy"



async def answer_with_rag(question: str, context: str) -> str:
    prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

Rules:
- Use only CONTEXT.
- If insufficient, output exactly: I don't know.
"""
    res = await answer_agent.run(task=prompt)
    for msg in reversed(res.messages):
        if getattr(msg, "source", None) == "rag_answerer" and hasattr(msg, "content"):
            return str(msg.content).strip()
    for msg in reversed(res.messages):
        if hasattr(msg, "content"):
            return str(msg.content).strip()
    return "I don't know."


async def generate_quiz_question() -> dict:
    history = get_recent_history(n=12)
    asked = get_asked_quiz_questions()

    user_questions = [h.get("user", "") for h in history if h.get("user")]

    prompt = f"""
recent_user_questions:
{json.dumps(user_questions, indent=2)}

already_asked_quiz_questions:
{json.dumps(asked, indent=2)}

Task:
Generate ONE new quiz question that:
1) Is conceptually related to recent_user_questions
2) Is NOT identical to any recent_user_questions
3) Is NOT semantically equivalent to any recent_user_questions
4) Is NOT identical to already_asked_quiz_questions
5) Tests deeper understanding of the topic

Return STRICT JSON only:
{{
  "quiz_question": "...",
  "topic": "...",
  "difficulty": "easy|medium|hard"
}}
"""
    return await _agent_json(quiz_gen_agent, prompt)



async def grade_quiz_answer(quiz_question: str, user_answer: str) -> dict:
    history = get_recent_history(n=8)
    topics = "\n".join([f"- {h.get('user','')}" for h in history])

    prompt = f"""Quiz question:
{quiz_question}

User answer:
{user_answer}

Recent topics:
{topics}

Return grading JSON only.
"""
    return await _agent_json(quiz_grade_agent, prompt)


def _reset_after_quiz() -> None:
    state.quiz_active = False
    state.quiz_question = None
    state.quiz_attempts = 0
    state.difficult_count = 0


def _reset_teach_mode() -> None:
    state.teach_active = False
    state.teach_stage = "none"
    state.teach_question = None


async def handle_user_message(user_text: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    """
    Returns payload for UI:
      {
        "answer": "...",
        "question_type": "simple|difficult" (for normal chat),
        "quiz_question": "..." (when quiz triggered),
      }
    """

    # ========================================================
    # TEACH MODE (highest priority, unless quiz is active)
    # ========================================================
    if state.teach_active and not state.quiz_active:
        # Stage 1: awaiting consent
        if state.teach_stage == "awaiting_consent":
            if _is_negative(user_text):
                append_chat_log({
                    "ts": now_utc_iso(),
                    "event": "teach_declined",
                    "teach_origin_question": state.teach_question,
                })
                _reset_teach_mode()
                return {"answer": "No problem. Ask me something else anytime.", "question_type": None}

            if _is_affirmative(user_text):
                state.teach_stage = "awaiting_content"
                return {
                    "answer": (
                        "Great — please teach me by either:\n"
                        "1) writing a short explanation (3–8 lines), OR\n"
                        "2) pasting key notes / bullet points.\n\n"
                        "If you only share a URL, I can’t browse it yet — paste the important content."
                    ),
                    "question_type": None,
                }

            return {
                "answer": "Please reply with **yes** to teach me, or **no** to skip.",
                "question_type": None,
            }

        # Stage 2: awaiting content
        if state.teach_stage == "awaiting_content":
            if _is_negative(user_text):
                append_chat_log({
                    "ts": now_utc_iso(),
                    "event": "teach_cancelled",
                    "teach_origin_question": state.teach_question,
                })
                _reset_teach_mode()
                return {"answer": "Okay, cancelled. Ask another question anytime.", "question_type": None}

            if _looks_like_url(user_text) and len(user_text.strip().split()) == 1:
                return {
                    "answer": (
                        "I can’t open links directly right now. Please paste the important content from that page "
                        "(or summarize it), and I’ll learn it."
                    ),
                    "question_type": None,
                }

            # Extract facts (safe)
            try:
                extracted = await _agent_json(
                    knowledge_extractor_agent,
                    f"User explanation:\n{user_text}\nReturn JSON only."
                )
            except Exception as e:
                _reset_teach_mode()
                return {"answer": f"⚠️ I couldn't extract facts safely. ({type(e).__name__}) Please try bullet points.", "question_type": None}

            facts = extracted.get("facts", [])
            if not isinstance(facts, list):
                facts = []

            if not facts:
                return {
                    "answer": (
                        "I couldn't extract clear factual chunks from that. "
                        "Can you rephrase with bullet points or short factual statements?"
                    ),
                    "question_type": None,
                }

            # Verify facts (safe)
            try:
                verified = await _agent_json(
                    knowledge_verifier_agent,
                    f"Facts:\n{facts}\nReturn JSON only."
                )
            except Exception as e:
                _reset_teach_mode()
                return {"answer": f"⚠️ Verification failed. ({type(e).__name__}) Knowledge not stored.", "question_type": None}

            valid = bool(verified.get("valid", False))
            cleaned_facts = verified.get("cleaned_facts", facts)
            if not isinstance(cleaned_facts, list):
                cleaned_facts = facts

            if not valid or not cleaned_facts:
                append_chat_log({
                    "ts": now_utc_iso(),
                    "event": "teach_rejected",
                    "teach_origin_question": state.teach_question,
                    "source_type": "user_provided",
                    "knowledge_added": False,
                })
                _reset_teach_mode()
                return {
                    "answer": (
                        "I’m not confident that explanation is consistent/accurate enough to store safely. "
                        "Knowledge not stored."
                    ),
                    "question_type": None,
                }

            # Upsert into Qdrant
            added_count = add_knowledge_to_qdrant(
                facts=[str(f).strip() for f in cleaned_facts if str(f).strip()],
                topic=state.teach_question or "unknown",
                source_text=user_text,
            )

            append_chat_log({
                "ts": now_utc_iso(),
                "event": "knowledge_added",
                "teach_origin_question": state.teach_question,
                "source_type": "user_provided",
                "knowledge_added": True,
                "facts_count": added_count,
            })

            _reset_teach_mode()
            return {
                "answer": f"✅ Thanks! I learned **{added_count}** new facts and saved them to my Qdrant knowledge base.",
                "question_type": None,
            }

        _reset_teach_mode()

    # ========================================================
    # QUIZ MODE
    # ========================================================
    if state.quiz_active and state.quiz_question:
        g = await grade_quiz_answer(state.quiz_question, user_text)
        state.quiz_attempts += 1

        correct = bool(g.get("correct", False))
        rationale = (g.get("rationale") or "").strip()
        ideal = (g.get("ideal_answer") or "").strip()

        if correct:
            append_quiz_log({
                "ts": now_utc_iso(),
                "event": "graded",
                "quiz_question": state.quiz_question,
                "user_answer": user_text,
                "correct": True,
                "rationale": rationale,
            })
            _reset_after_quiz()
            return {"answer": f"✅ Correct.\n\nRationale: {rationale}", "question_type": None}

        if state.quiz_attempts < QUIZ_MAX_ATTEMPTS:
            return {"answer": f"❌ Not quite.\n\nRationale: {rationale}\n\nTry again (second attempt):", "question_type": None}

        append_quiz_log({
            "ts": now_utc_iso(),
            "event": "graded",
            "quiz_question": state.quiz_question,
            "user_answer": user_text,
            "correct": False,
            "rationale": rationale,
            "ideal_answer": ideal,
        })
        _reset_after_quiz()
        return {"answer": f"❌ Incorrect.\n\nRationale: {rationale}\n\n✅ Correct answer: {ideal}", "question_type": None}

    # ========================================================
    # NORMAL MODE (RAG)
    # ========================================================
    difficulty = await grade_difficulty(user_text)
    question_type = "simple" if difficulty == "easy" else "difficult"

    passages = retrieve_top_k(user_text, top_k=top_k)
    context = format_context(passages) if passages else "No relevant context found."
    bot_answer = await answer_with_rag(user_text, context)

    retrieved_any_text, sources, top_score = retrieval_metadata(passages)

    append_chat_log({
        "ts": now_utc_iso(),
        "user": user_text,
        "bot": bot_answer,
        "difficulty": difficulty,
        "top_k": top_k,
        "retrieved_any_text": retrieved_any_text,
        "sources": sources,
        "top_score": top_score,
    })

    # ---------------------------
    # 1️⃣ Increment difficult counter (session-wide)
    # ---------------------------
    if difficulty == "difficult":
        state.difficult_count += 1

    # ---------------------------
    # 2️⃣ Trigger quiz if >= 3 difficult in session
    # ---------------------------
    quiz_question = None
    if state.difficult_count >= DIFFICULTY_TRIGGER_COUNT:
        q = await generate_quiz_question()
        quiz_question = (q.get("quiz_question") or "").strip()

        if quiz_question:
            append_quiz_log({
                "ts": now_utc_iso(),
                "event": "asked",
                "quiz_question": quiz_question,
                "topic": q.get("topic", ""),
                "difficulty": q.get("difficulty", ""),
            })

            state.quiz_active = True
            state.quiz_question = quiz_question
            state.quiz_attempts = 0

            return {
                "answer": bot_answer,
                "question_type": question_type,
                "quiz_question": quiz_question,
            }

    # ---------------------------
    # 3️⃣ THEN check teach-mode
    # ---------------------------
    if bot_answer.strip() == "I don't know.":
        state.teach_active = True
        state.teach_stage = "awaiting_consent"
        state.teach_question = user_text

        return {
            "answer": "I don't know.\n\nWould you like to teach me this concept? (yes/no)",
            "question_type": question_type,
        }

    return {"answer": bot_answer, "question_type": question_type}


# ============================================================
# Persistent Async Loop Runner (fixes "event loop is closed")
# ============================================================
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: int = 300):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)


async_runner = AsyncLoopThread()


def run_async(coro):
    try:
        return async_runner.run(coro, timeout=300)
    except FuturesTimeoutError:
        raise TimeoutError("Agent call timed out.")


# ============================================================
# Flask routes
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", messages=get_ui_messages())


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"answer": "⚠️ Empty message.", "question_type": None})

    # Update UI message history (user)
    msgs = get_ui_messages()
    msgs.append({"role": "user", "content": user_msg})
    set_ui_messages(msgs)

    # Run agent
    try:
        result = run_async(handle_user_message(user_msg, top_k=TOP_K_DEFAULT))
    except Exception as e:
        result = {"answer": f"⚠️ Server error: {type(e).__name__}: {e}", "question_type": None}

    # Append assistant response to UI messages
    msgs = get_ui_messages()
    msgs.append({"role": "assistant", "content": result.get("answer", "")})

    # Store question_type on the user message for badge rendering on refresh
    if result.get("question_type"):
        if len(msgs) >= 2 and msgs[-2].get("role") == "user":
            msgs[-2]["question_type"] = result["question_type"]

    set_ui_messages(msgs)
    return jsonify(result)


@app.route("/clear", methods=["POST"])
def clear():
    set_ui_messages([])

    # Reset state
    state.difficult_count = 0
    state.quiz_active = False
    state.quiz_question = None
    state.quiz_attempts = 0
    _reset_teach_mode()

    return jsonify({"ok": True})


if __name__ == "__main__":
    # Disable reloader to avoid double-process + event loop issues
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
