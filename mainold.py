from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables (Make sure your GEMINI_API_KEY is in your .env file!)
load_dotenv()

app = FastAPI(title="CRM Copilot Backend")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./crm_db")
collection = chroma_client.get_or_create_collection(name="lead_notes")

# Initialize Gemini
ai_client = genai.Client()

class crmRequest(BaseModel):
    lead_email: str
    user_prompt: str

@app.post("/api/copilot")
async def crm_copilot(request: crmRequest):
    print(f"\n--- Searching memory for {request.lead_email}... ---")
    
    # 1. RAG RETRIEVAL: Search the database
    results = collection.query(
        query_texts=[request.user_prompt],
        n_results=2,
        where={"email": request.lead_email}
    )
    retrieved_notes = results['documents'][0] if results['documents'] else []

    # 2. RAG AUGMENTATION: Build the super-prompt
    # We turn the array of notes into a single block of text
    context_string = "\n- ".join(retrieved_notes) 
    
    augmented_prompt = f"""
    You are an AI Copilot for a CRM system. 
    Answer the sales rep's question about their lead using ONLY the notes provided below.
    Be concise, helpful, and professional.
    If the answer is not in the notes, say "I don't have enough information in the CRM to answer that."
    
    Lead Email: {request.lead_email}
    
    CRM Notes:
    - {context_string}
    
    Sales Rep Question: {request.user_prompt}
    """

    print("--- Asking Gemini to synthesize the answer... ---")

    # 3. RAG GENERATION: Send it to the LLM
    response = ai_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=augmented_prompt
    )

    # 4. Return the clean, AI-generated answer to Laravel!
    return {
        "status": "success",
        "lead_email": request.lead_email,
        "ai_answer": response.text,
        "sources_used": retrieved_notes
    }

# seed_database() # <-- Commented out because the data is already saved to your hard drive!