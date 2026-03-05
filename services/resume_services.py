import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from llm_factory.get_llm import get_gemini_llm
from embedding_factory.get_embedding_model import get_embedding_model
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

def get_vector_store():
    embedding_model = get_embedding_model()
    if os.path.exists(VECTOR_STORE_DIR):
        return Chroma(persist_directory=VECTOR_STORE_DIR,embedding_function=embedding_model)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    return Chroma(persist_directory=VECTOR_STORE_DIR,embedding_function=embedding_model
    )

vectorstore = get_vector_store()

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


def store_resume_analysis(analysis, doc_id):
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


def analyze_resume(job_requirements, resume_text):

    llm = get_gemini_llm()

    prompt = PromptTemplate(
        input_variables=["job_requirements", "resume_text"],
        template="""
You are an expert HR and recruitment specialist.

Analyze the resume against the job requirements.

Job Requirements:
{job_requirements}

Resume:
{resume_text}

Provide a structured analysis explaining how well the resume matches the job.

At the end include:

Suitability Score: XX%
"""
    )

    chain = (
        RunnableMap({
            "job_requirements": lambda x: x["job_requirements"],
            "resume_text": lambda x: x["resume_text"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({
        "job_requirements": job_requirements,
        "resume_text": resume_text
    })

    return result