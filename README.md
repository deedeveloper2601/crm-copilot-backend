# VectorLead AI - CRM Copilot Microservice

An asynchronous Python microservice built with **FastAPI** that acts as an intelligent AI Copilot for CRM systems. 

This service is designed to handle heavy machine learning workloads—such as local vector embeddings, semantic search, and Large Language Model (LLM) orchestration—keeping them strictly isolated from the primary web application (e.g., a Laravel monolith).

## 🚀 Features

* **Retrieval-Augmented Generation (RAG):** Instantly searches and synthesizes thousands of customer records and call transcripts to provide accurate, context-aware AI responses.
* **Local Embeddings:** Utilizes Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) to generate 384-dimensional math arrays locally on the CPU, eliminating third-party embedding API costs.
* **Vector Database:** Engineered with **PostgreSQL** and the `pgvector` extension to seamlessly join traditional relational CRM data with high-dimensional AI arrays.
* **Autonomous LLM Tool Calling:** Integrated with the **Google Gemini API**. The AI agent can read user intent and autonomously trigger backend database mutations (like logging calls, saving notes, or updating lead statuses) purely through natural language.

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python)
* **AI/LLM:** Google Gemini API (`google-genai`)
* **Embeddings:** Hugging Face Sentence Transformers
* **Database:** PostgreSQL + `pgvector`
* **ORM:** SQLAlchemy
* **Server:** Uvicorn

---

## 💻 Installation & Setup

### 1. Prerequisites
* Python 3.9+
* Docker (for running PostgreSQL with the `pgvector` extension)
