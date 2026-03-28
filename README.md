\# Tesla 10-K RAG Chatbot



A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about Tesla’s 10-K report.



\## Features

\- Upload and process Tesla 10-K PDF

\- Uses embeddings for document search

\- Retrieves relevant context

\- Generates answers using Gemini LLM



\## Tech Stack

\- Python

\- LangChain

\- Google Gemini API

\- Pinecone (Vector Database)

\- HuggingFace Embeddings



\## Project Structure



tesla-rag/

│

├── ingest.py        # Load and embed Tesla 10K

├── chat.py          # Chat interface

├── requirements.txt

├── .gitignore

└── README.md



\## Setup



Clone the repo:



```bash

git clone https://github.com/Subhra-Nandi/tesla-rag.git

cd tesla-rag

