from langchain_google_genai import GoogleGenerativeAI
from config.settings import ProjectSettings

settings = ProjectSettings()
GOOGLE_API_KEY=settings.GOOGLE_API_KEY


_current_model_name = None
_current_llm_instance = None

def get_gemini_llm(model_name: str = "gemini-2.5-flash"):
    global _current_model_name, _current_llm_instance
    if _current_model_name == model_name and _current_llm_instance is not None:
        return _current_llm_instance
    print(f"Model Name = {model_name}")
    llm = GoogleGenerativeAI(model=f"models/{model_name}",api_key=GOOGLE_API_KEY)
    _current_model_name = model_name
    _current_llm_instance = llm
    return llm

# Example usage
#check_llm = get_gemini_llm(model_name="gemini-2.5-flash")
#print(check_llm)
#print(type(check_llm))