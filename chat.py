import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

#  Settings
index_name = "tesla-rag"

#  Embeddings (must match ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#  Connect to Pinecone
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#  Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3
)

#  Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
""")

#  RAG Chain (Modern Style)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

#  Chat Loop
while True:
    query = input("\nAsk about Tesla 10K (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    result = rag_chain.invoke(query)
    print("\nAnswer:\n")
    print(result)