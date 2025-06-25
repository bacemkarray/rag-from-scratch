# RAG From Scratch

This is a **minimal RAG (Retrieval-Augmented Generation)** prototype I built while following [this tutorial](https://github.com/PradipNichite/Youtube-Tutorials/tree/main/Vanilla%20RAG) to understand how tools like ChromaDB, sentence-transformers, and an LLM work together in a local pipeline.

## 🧠 Why I Did This

I’ve used AI tools before, but this was my first time breaking things down at a systems level - no external orchestration frameworks, just raw file parsing, embedding, vector search, and response generation. I wanted a behind-the-scenes look at how RAG works under the hood.

This project also gave me a chance to:
- Practice **basic modularity** across multiple Python files
- Explore how conversation memory, query rewriting, and LLM prompting fit into a cohesive loop
- Understand **why abstraction layers like LangChain exist** — not just for convenience, but to handle real architectural complexity that would otherwise get unwieldy at scale

## 🛠️ What's Inside

- `app.py` -  Orchestrates communication between the user, retriever, and LLM.
- `conversation_memory.py` - Maintains conversation history throughout a session (Required since this app uses the OpenAI Chat Completions API not Responses API).
- `query_refiner.py` - Converts follow-up questions into standalone queries using chat history.
- `retrieval.py` - Performs semantic search over embedded document chunks in ChromaDB.
- `llm_client.py` - Initializes the LLM client and formats prompts.
- `utils.py` - Handles document parsing, chunking, and batching.

![Code structure preview](https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Vanilla%20RAG/rag%20flowchart%20new.png)

## 🚧 Next Steps

I don’t plan to build larger LLM applications entirely from scratch, as doing so would introduce unnecessary complexity. Now that I’ve seen how the parts fit together, I plan to:

- Use frameworks like **LangChain** or **LangGraph** to manage multi-step LLM workflows
- Explore **agent-style orchestration** using state machines or tool calling
- Apply what I’ve learned in **more ambitious personal projects**, where abstraction is a feature, not a crutch

This project served its purpose: to provide me with a clear mental model of the RAG building blocks, allowing me to move forward with better judgment and more modular design.

