import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config.settings import ProjectSettings

proj_settings = ProjectSettings()
VECTOR_STORE = proj_settings.VECTOR_STORE
DOCUMENTS_DIR = proj_settings.DOCUMENTS_DIR

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

DOCUMENTS_DIR = os.path.join(working_dir, DOCUMENTS_DIR)
VECTOR_STORE_DIR = os.path.join(working_dir, VECTOR_STORE)
temp_file_path = os.path.join(working_dir, "temp.txt")

def initialize_vector_store(embedding_model):
    """
    Create or load Chroma vector store
    """
    if os.path.exists(VECTOR_STORE_DIR):
        vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR,embedding_function=embedding_model)
    else:
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR,embedding_function=embedding_model)

    return vectorstore


def extract_text_from_resume(file):
    """
    Extract text from uploaded resume file
    Supports PDF, DOCX and TXT
    """
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)

        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)

        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])

        return text
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def split_text(text):
    """
    Split text into smaller chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    return splitter.create_documents([text])


def store_resume_analysis(vectorstore, analysis, doc_id):
    """
    Store resume analysis in vector database
    """
    documents = split_text(analysis)
    vectorstore.add_documents(
        documents,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))]
    )
    vectorstore.persist()


def extract_suitability_score(text):
    """
    Extract Suitability Score from LLM response
    """
    match = re.search(r"Suitability Score: (\d{1,3})%", text)

    if match:
        return int(match.group(1))

    return None


def analyze_resume():
    return None