from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


def get_Gemini(name: str = "deepseek-r1-distill-llama-70b"):
    """Gemini 2.0 Flash	15	1,000,000	1,500"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
    except Exception as e:
        print(f"Error get api key Add Gemini API Key {e}")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=api_key,
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm
    except Exception as e:
        print("Error init LLM : ", e)
        return None


def get_groq_llm(name: str = "deepseek-r1-distill-llama-70b"):
    """qwen-qwq-32b	15	1,000,000	1,500"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
    except Exception as e:
        print(f"Error get api key Add Groq API Key {e}")
        return None
    try:
        llm = ChatGroq(
            temperature=0.8,
            api_key=str(api_key),
            model_name=name
        )

        return llm
    except Exception as e:
        print("Error init LLM : ", e)
        return None
