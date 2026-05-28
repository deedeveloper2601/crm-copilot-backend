from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# --- Postgres & SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# Load environment variables (Make sure your GEMINI_API_KEY is in your .env file)
load_dotenv()

app = FastAPI(title="VectorLead AI - CRM Copilot Backend")

# 1. Initialize the Local Math Embedder
print("Loading local AI embedder...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Setup PostgreSQL Connection (Running on port 5433 from our Docker setup)
DATABASE_URL = "postgresql://postgres:secret@localhost:5433/postgres"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 3. Define our Database Model
class CrmNote(Base):
    __tablename__ = "crm_notes"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_email = Column(String, index=True) 
    note_text = Column(Text)
    embedding = Column(Vector(384)) # The AI math array

# 4. Create the table and enable the vector extension
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(bind=engine)

# 5. Initialize Gemini
ai_client = genai.Client()

# --- Seed the Postgres Database (Runs once when the server starts) ---
def seed_postgres():
    db = SessionLocal()
    if db.query(CrmNote).first() is None:
        notes = [
            {"email": "digambar@test.com", "text": "Had a great call with Digambar today. He is very interested in the Pro plan."},
            {"email": "digambar@test.com", "text": "Digambar wants a follow-up call next Tuesday at 2 PM to finalize the deal."},
            {"email": "john@test.com", "text": "John is complaining about the UI."}
        ]
        
        for note in notes:
            vector_math = embedder.encode(note["text"]).tolist()
            new_record = CrmNote(
                lead_email=note["email"],
                note_text=note["text"],
                embedding=vector_math
            )
            db.add(new_record)
        db.commit()
        print("✅ Fake CRM data loaded into PostgreSQL!")
    db.close()

seed_postgres()

# --- The AI Tool (Database Writer) ---
def add_crm_note(lead_email: str, note_text: str) -> str:
    """
    Logs a new call, interaction, status update, or note for a specific lead in the CRM.
    Use this tool when the user explicitly asks to save a note, log a call, or update a lead's status.
    """
    print(f"🛠️ Tool Triggered: Saving new note for {lead_email}...")
    db = SessionLocal()
    
    try:
        # Convert the new note to math so it is instantly searchable later
        vector_math = embedder.encode(note_text).tolist()
        
        # Save it to Postgres
        new_record = CrmNote(
            lead_email=lead_email,
            note_text=note_text,
            embedding=vector_math
        )
        db.add(new_record)
        db.commit()
        return "Success: The note was successfully saved to the CRM database."
    except Exception as e:
        db.rollback()
        return f"Error: Could not save the note. {str(e)}"
    finally:
        db.close()

# --- Incoming Request Model ---
class crmRequest(BaseModel):
    lead_email: str
    user_prompt: str

# --- The Main API Endpoint ---
# Note: We use 'def' instead of 'async def' so the heavy math doesn't freeze the server
@app.post("/api/copilot")
def crm_copilot(request: crmRequest):
    print(f"\n--- Searching Postgres for {request.lead_email}... ---")
    
    db = SessionLocal()
    
    # 1. RAG: Convert the user's search prompt into math
    search_vector = embedder.encode(request.user_prompt).tolist()
    
    # 2. Query Postgres for the closest matching notes
    results = db.query(CrmNote.note_text).filter(
        CrmNote.lead_email == request.lead_email
    ).order_by(
        CrmNote.embedding.cosine_distance(search_vector)
    ).limit(2).all()
    
    db.close()

    # 3. Format the retrieved notes into a string
    retrieved_notes = [str(record[0]) for record in results]
    context_string = "\n- ".join(retrieved_notes) 

    print("--- Asking Gemini to process the request... ---")

    # 4. Construct the prompt for Gemini
    augmented_prompt = f"""
    You are VectorLead AI, a Copilot for a CRM system. 
    You have two jobs:
    1. Answer questions about the lead using ONLY the provided CRM Notes below.
    2. Log new notes or updates if the sales rep explicitly asks you to, using your provided tools.
    
    Be concise, helpful, and professional.

    Lead Email: {request.lead_email}
    
    Existing CRM Notes context:
    - {context_string}
    
    Sales Rep Input: {request.user_prompt}
    """

    # 5. Call Gemini and pass it the Tool
    response = ai_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=augmented_prompt,
        config={"tools": [add_crm_note]} # This enables the database writing!
    )

    # Note: Because Gemini might use a tool, it doesn't always return standard text immediately.
    # The SDK handles the tool call execution under the hood and returns the final text response.
    
    return {
        "status": "success",
        "lead_email": request.lead_email,
        "ai_answer": response.text,
        "sources_used": retrieved_notes
    }

