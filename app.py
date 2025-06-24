from retrieval import semantic_search, get_context_with_sources, collection
from query_refiner import contextualize_query
from conversation_memory import format_history_for_prompt, add_message, create_session
from llm_client import generate_response, client

def conversational_rag_query(collection, raw_query: str, session_id: str, n_chunks: int = 3):
    """Perform RAG query with conversation history"""
    # Get conversation history
    conversation_history = format_history_for_prompt(session_id)

    # Handle follow up questions
    refined_query = contextualize_query(raw_query, conversation_history, client)

    # Get relevant chunks
    context, sources = get_context_with_sources(
        semantic_search(collection, refined_query, n_chunks)
    )

    response = generate_response(refined_query, context, conversation_history)

    # Add to conversation history
    add_message(session_id, "user", refined_query)
    add_message(session_id, "assistant", response)

    return response, sources


if __name__ == "__main__":
    session = create_session()
    # load or create your chroma collection…
    while True:
        q = input("You: ")
        resp, src = conversational_rag_query(collection, q, session)
        print("AI:", resp)
        print("Sources:", src)