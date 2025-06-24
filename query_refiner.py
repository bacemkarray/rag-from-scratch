from openai import OpenAI

def contextualize_query(query: str, conversation_history: str, client: OpenAI):
    """Convert follow-up questions into standalone queries"""
    contextualize_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone 
    question which can be understood without the chat history. Do NOT answer 
    the question, just reformulate it if needed and otherwise return it as is."""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": contextualize_prompt},
                {"role": "user", "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error contextualizing query: {str(e)}")
        return query  # Fallback to original query