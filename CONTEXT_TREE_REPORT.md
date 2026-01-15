# ContextTree Codebase Report

## 1. Project Overview (What We Do)
**ContextTree** (also referred to as "Convo Canvas API") is a backend API for a sophisticated chat application. Its primary purpose is to facilitate user-AI conversations with advanced structural capabilities.

*   **Core Functionality:** It powers a chat interface where users can converse with an AI.
*   **Unique Selling Point:** It supports **Thread Forking**, allowing users to branch off a conversation from a specific message point, creating alternate conversation paths while preserving context.
*   **AI Engine:** Uses Large Language Models (LLMs) such as Groq and Gemini, orchestrated via `LangChain` and `LangGraph`.

## 2. Architecture & Implementation (How)
The system is built using a modern Python web stack:

*   **Web Framework:** **FastAPI** serves as the backbone, handling HTTP requests, routing, and middleware.
*   **Agent Orchestration:** **LangGraph** is used to define the conversational flow as a state machine. The graph manages nodes like `assistant` and `summurize` (sic), handling the logic of when to reply or when to summarize context.
*   **Data Models:** **Pydantic** is used for robust data validation on request/response bodies (`app/schemas`).
*   **Rate Limiting:** **SlowAPI** is implemented to prevent abuse (e.g., limiting chat requests to 5 per minute).

### Key Workflows
*   **Chat:** The `/api/v1/chat` endpoint invokes the LangGraph `get_response` method. The graph processes the message, potentially using tools or strictly LLM inference, and returns the response.
*   **Forking:** The `/api/v1/fork` endpoint interacts directly with the `MongoConversationStore` to duplicate thread history up to a specific message ID, effectively creating a new branch.

## 3. Data Storage (Where We Store)
The application uses a hybrid storage approach:

1.  **MongoDB (Primary Storage):**
    *   **Purpose:** Persists conversation history, user data, and vector embeddings.
    *   **Collections:**
        *   `users`: Stores user documents and their thread references.
        *   `messages`: A flat collection storing full message content and embeddings.
    *   **Search:** Uses **Atlas Vector Search** (`embedding_vector_index`) for semantic retrieval.
    *   **Implementation:** `app/agent/store/MongoStore.py`.

2.  **Redis (State Management):**
    *   **Purpose:** Acts as a **checkpointer** for `LangGraph`. It saves the state of the conversation graph (active node, variables) so conversations can continue across multiple stateless HTTP requests.
    *   **Implementation:** `langgraph.checkpoint.memory.MemorySaver` (configured via `redis_saver`).

## 4. Strengths (Good)
*   **Modern Technology Stack:** Utilization of `LangGraph` and `FastAPI` puts the project on the cutting edge of Python AI development.
*   **Scalability Design:** The use of async API endpoints and separate storage layers (Redis/Mongo) suggests a design ready for concurrency.
*   **Modular Structure:** The codebase is well-organized into `agent`, `api`, `core`, and `schemas`, making navigation intuitive.
*   **Advanced Features:** The "Forking" capability is a significant differentiator from standard linear chat applications.
*   **Safety Measures:** Rate limiting is active on public endpoints.

## 5. Flaws & Weaknesses
*   **Hardcoded Configuration:**
    *   `app/agent/store/MongoStore.py` contains a hardcoded MongoDB Atlas connection string template. This tightly couples the code to a specific cloud cluster structure.
    *   `app/agent/helpers/get_llm.py` hardcodes model names (`gemini-2.0-flash`, `deepseek-r1...`) and parameters.
*   **Code Quality & Typography:**
    *   There are spelling errors in key logical components (e.g., `summurize` instead of `summarize` in `app/agent/main.py`).
    *   Commented-out code (dead code) is present in production files (e.g., `InMemorySaver` usage).
*   **Error Handling:**
    *   In `app/agent/nodes/assistant_node.py`, generic `Exception` blocks print errors to stdout (`print("Error in assistant node:", e)`). This should be logged via the proper logger and potentially raised so the API can return a 500 error instead of failing silently or returning a partial state.

## 6. Potential Pitfalls
*   **Vendor Lock-In:** The MongoDB implementation relies specifically on Atlas Search (`create_search_index`). Migrating to a self-hosted MongoDB or another vector store would require significant refactoring.
*   **State Desynchronization:** If the Redis checkpoint and MongoDB storage get out of sync (e.g., a message is saved to Mongo but the graph state isn't updated in Redis due to a crash), the conversation flow could break.
*   **Environment Dependency:** The heavy reliance on `.env` without validation in some parts (like `get_llm.py` returning `None` if keys are missing) can lead to runtime `AttributeError`s later in the execution flow.

## 7. Recommended Improvements
1.  **Externalize Connection Strings:** Move the full MongoDB connection URI construction to `app/core/config.py` and rely solely on the environment variable, removing the hardcoded Atlas string.
2.  **Robust Error Handling:** actively use `app.core.logger` everywhere. Remove `print()` statements. Ensure critical failures in the graph bubble up to the API layer.
3.  **Spelling Fixes:** Correct `summurize` to `summarize` in `main.py` and `assistant_node.py` to maintain code professionalism.
4.  **Unit Testing:** Create a test suite for the `MongoConversationStore` to verify forking logic handles edge cases (e.g., forking from non-existent message) correctly.
