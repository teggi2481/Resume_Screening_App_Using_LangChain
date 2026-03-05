from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class ProjectSettings(BaseSettings):

    OLLAMA_URL: str
    GOOGLE_API_KEY: str
    LLM_MODELS: str
    DOCUMENTS_DIR: str
    VECTOR_STORE: str
    SENTENCE_WINDOW_SIZE: int = 3
    DEVICE: str = 'cpu'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"