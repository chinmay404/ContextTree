from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv() 

def get_embedding(querry :str):
    
    try:
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
        # result = embeddings.embed_query(querry)
        # return result
        return [0.0] * 1536
    except Exception as e:
        print(f"Error in embedding: {e}")
        return [0.0] * 1536