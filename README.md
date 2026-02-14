# 🧠 Adaptive RAG Tutor with Quiz & Teach Mode

An advanced **Retrieval-Augmented Generation (RAG)** tutoring system
built with:

-   **Flask** (Web API + UI)
-   **Ollama** (LLM + Embeddings)
-   **Qdrant** (Vector Database)
-   **AutoGen Agents** (Multi-agent orchestration)

This system dynamically:

-   Classifies question difficulty
-   Uses strict RAG-based answering
-   Triggers quizzes after difficult questions
-   Allows users to *teach the system* when it doesn't know something
-   Stores verified user-provided knowledge into Qdrant

------------------------------------------------------------------------

# 🚀 Features

## 1️⃣ Strict RAG Answering

-   Uses embedding search via Qdrant
-   Answers ONLY using retrieved context
-   Returns `"I don't know."` if context is insufficient
-   No hallucinations allowed

## 2️⃣ Automatic Difficulty Detection

-   Questions are classified as:
    -   `easy`
    -   `difficult`
-   After **3 difficult questions**, a quiz is triggered

## 3️⃣ Adaptive Quiz Mode

-   Generates quiz questions based on recent user topics
-   Prevents repeated questions
-   Allows 2 attempts
-   Grades answers using an LLM agent
-   Logs results to `quiz_log.json`

## 4️⃣ Teach Mode (Self-Learning System)

If the model responds with:

    I don't know.

It will ask:

    Would you like to teach me this concept? (yes/no)

If user agrees:

1.  Extracts factual statements
2.  Verifies consistency and correctness
3.  Embeds verified facts
4.  Stores them in Qdrant

This creates a **continuously improving knowledge base**.

------------------------------------------------------------------------

# 🏗 Architecture Overview

User → Flask → Difficulty Agent → Retrieval (Qdrant) → RAG Answer Agent\
                                             ↓\
                                     Quiz Generator / Grader\
                                             ↓\
                                     Teach Mode Agents

Agents used:

-   `difficulty_grader`
-   `rag_answerer`
-   `quiz_generator`
-   `quiz_grader`
-   `knowledge_extractor`
-   `knowledge_verifier`

All agents share a single persistent Ollama model client.

------------------------------------------------------------------------

# 🧩 Tech Stack

-   Python 3.10+
-   Flask
-   Ollama
-   Qdrant
-   AutoGen AgentChat
-   Qwen Models:
    -   `qwen3:1.7b`
    -   `qwen3-embedding:4b`

------------------------------------------------------------------------

# ⚙️ Installation

## 1️⃣ Install Dependencies

``` bash
pip install flask ollama qdrant-client autogen-agentchat autogen-ext
```

## 2️⃣ Start Ollama

``` bash
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:4b
ollama serve
```

## 3️⃣ Start Qdrant

Using Docker:

``` bash
docker run -p 6333:6333 qdrant/qdrant
```

## 4️⃣ Configure Environment

Optional:

``` bash
export FLASK_SECRET_KEY="your-secret-key"
```

## 5️⃣ Run the App

``` bash
python app.py
```

App runs at:

    http://localhost:5000

------------------------------------------------------------------------

# 📁 Project Structure

    .
    ├── app.py
    ├── templates/
    │   └── index.html
    ├── chat_log.json
    ├── quiz_log.json
    └── README.md

------------------------------------------------------------------------

# 🧠 Retrieval System

-   Embeddings generated via Ollama
-   Stored in Qdrant collection: `machine-learning`
-   Top-K retrieval (default = 10)
-   Metadata logging includes:
    -   source
    -   score
    -   top_score
    -   retrieval success flag

------------------------------------------------------------------------

# 📊 Logging

## chat_log.json

Stores:

-   user message
-   bot response
-   difficulty
-   retrieval metadata
-   timestamps

## quiz_log.json

Stores:

-   quiz questions asked
-   user answers
-   correctness
-   rationales
-   ideal answers

------------------------------------------------------------------------

# 🔐 Safety & Guardrails

-   Strict JSON extraction for agents
-   Verification before storing user knowledge
-   No external knowledge allowed in RAG agent
-   Event loop isolation to prevent async crashes
-   Disabled Flask reloader to prevent duplicate loops

------------------------------------------------------------------------

# 🔄 Event Loop Architecture

The system runs a persistent async event loop in a background thread:

``` python
class AsyncLoopThread:
```

This prevents:

-   `RuntimeError: Event loop is closed`
-   Multi-request async conflicts
-   Model client reinitialization issues

------------------------------------------------------------------------

# 📈 Future Improvements

-   Per-user session isolation (currently global state)
-   URL content ingestion
-   Knowledge conflict resolution
-   Vector deduplication checks
-   Web UI improvements
-   Docker Compose setup

------------------------------------------------------------------------

# 🧑‍💻 Author

Built as an adaptive AI tutor combining:

-   RAG
-   Multi-agent orchestration
-   Human-in-the-loop learning
-   Self-improving vector memory

------------------------------------------------------------------------

# 📜 License

MIT License (recommended)

------------------------------------------------------------------------

# ⭐ If You Like This Project

Give it a star on GitHub and experiment with:

-   Different embedding models
-   Larger LLMs
-   Different quiz strategies
-   Advanced retrieval pipelines

------------------------------------------------------------------------
