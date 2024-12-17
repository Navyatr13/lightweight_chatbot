from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.retriever import load_vector_store, retrieve_context
from app.response_generator import generate_response
from app.config import VECTOR_DB_PATH
import warnings
warnings.filterwarnings("ignore")

MAX_CONTEXT_LENGTH = 350
app = FastAPI()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("frontend/favicon.ico")

    
@app.post("/query")
def query_chatbot(request: QueryRequest):
    query = request.query
    print(f"Received query: {query}")

    # Retrieve relevant context
    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context(query, vector_store)
    if not context or len(context.strip()) == 0:
        context = "The user has initiated a conversation."
    print(f"Retrieved context: {context}")

    # Truncate context if too long
    if len(context.split()) > MAX_CONTEXT_LENGTH:
        context = " ".join(context.split()[:MAX_CONTEXT_LENGTH])

    # Generate response
    response = generate_response(query, context)
    print(f"Generated response: {response}")

    # Append response to conversation history
    conversation_history.append(f"User: {query}")
    conversation_history.append(f"Bot: {response}")
    print(f"Updated conversation history: {conversation_history}")

    return {"conversation": conversation_history[-6:]}


