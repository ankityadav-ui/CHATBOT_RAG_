#read files 
#break in chunks
#create embeddings
#store in vextor store ->FAISS

import os
from dotenv import load_dotenv

# Document Loaders
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Stores
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

DATA_PATH = "../knowledge_base"
FAISS_PATH = "../faiss_index"

print("Loading text files ...")

# Loader for text files
txt_loader = DirectoryLoader(
    DATA_PATH, 
    glob="**/*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={"encoding": "utf-8"}
)
txt_docs = txt_loader.load()

# Loader for PDF files
pdf_loader = DirectoryLoader(
    DATA_PATH, 
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader
)
pdf_docs = pdf_loader.load()

docs = txt_docs + pdf_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap =150)
docs = text_splitter.split_documents(docs)

print("Creating embeddings (this may take a moment)...")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Vector Store
db = FAISS.from_documents(docs, embeddings)

# Save the index locally
db.save_local(FAISS_PATH)

print("Embeddings and FAISS index created and saved successfully.")

