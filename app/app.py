from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from app.retriever import build_vector_store, load_vector_store, retrieve_context
from app.config import DOCUMENT_PATH, VECTOR_DB_PATH
from app.response_generator import generate_response

MAX_CONTEXT_LENGTH = 150 
# Define the FastAPI app
app = FastAPI()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

# Mount the static directory to serve index.html and related files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# In-memory conversation history (for simplicity)
conversation_history: List[str] = []

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    # Serve the index.html from the static folder
    with open("frontend/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/query")
def query_chatbot(request: QueryRequest):
    query = request.query

    # Vector retrieval
    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context(query, vector_store)
    if len(context.split()) > MAX_CONTEXT_LENGTH:
        context = " ".join(context.split()[:MAX_CONTEXT_LENGTH])

    # Response generation
    response = generate_response(query, context)
    # Return response
    conversation_history.append(response)
        
    return {"conversation": conversation_history[-6:]}


