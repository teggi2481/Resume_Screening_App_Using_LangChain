import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import ProjectSettings

# Load environment variables
load_dotenv()

def get_embedding_model():
    """
    Factory function to initialize and return the embedding model.
    This keeps embedding configuration centralized.
    """
    proj_settings = ProjectSettings()
    GOOGLE_API_KEY = proj_settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables or settings.")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    return embedding_model