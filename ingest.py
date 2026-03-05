import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

#  Pinecone settings
index_name = "tesla-rag"

#  Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#  Load PDF (since it's in root folder)
loader = PyPDFLoader("report.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF")

#  Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks")

#  Create embeddings (384 dimension)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name
)

print(" Data successfully stored in Pinecone!")