from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env into os.environ
api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def get_prompt(context, conversation_history, query):
  prompt = f"""Based on the following context and conversation history, please provide a relevant and contextual response.
    If the answer cannot be derived from the context, only use the conversation history or say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""
  return prompt


# Updated generate response function with conversation history also passed for Chatbot Memory
def generate_response(query: str, context: str, conversation_history: str = ""):
    """Generate a response using OpenAI with conversation history"""
    prompt = get_prompt(context, conversation_history, query)
    # print(prompt)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response.choices[0].message.content