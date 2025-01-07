import sys
import os
# Get the absolute path of the `app` folder
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(app_dir)
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from retriever import load_vector_store, retrieve_context
from response_generator import generate_response
from config import VECTOR_DB_PATH
from twilio.twiml.messaging_response import MessagingResponse

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
    start_time = time.time()
    query = request.query
    print(f"Received query: {query}")

    # Retrieve relevant context
    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context(query, vector_store)
    if not context or len(context.strip()) == 0:
        context = "The user has initiated a conversation."
    # Truncate context if too long
    if len(context.split()) > MAX_CONTEXT_LENGTH:
        context = " ".join(context.split()[:MAX_CONTEXT_LENGTH])

    # Generate response
    response = generate_response(query, context)
    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Time taken for retrieval and response: {total_time:.2f} seconds")
    print(f"Generated response: {response}")

    # Append response to conversation history
    conversation_history.append(f"User: {query}")
    conversation_history.append(f"Bot: {response}")
    return {"conversation": conversation_history[-6:]}


@app.post("/sms")
async def sms_interaction(request: Request):
    """Handles incoming SMS from Twilio."""
    form_data = await request.form()
    query = form_data.get("Body")  # Extract SMS body
    print(f"Received SMS query: {query}")

    start_time = time.time()

    # Retrieve context and generate response
    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context(query, vector_store)
    if not context or len(context.strip()) == 0:
        context = "The user has initiated a conversation."
    if len(context.split()) > MAX_CONTEXT_LENGTH:
        context = " ".join(context.split()[:MAX_CONTEXT_LENGTH])
    response = generate_response(query, context)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken for SMS response: {total_time:.2f} seconds")
    print(f"Generated SMS response: {response}")

    # Prepare Twilio MessagingResponse
    twilio_response = MessagingResponse()
    twilio_response.message(response)

    return PlainTextResponse(str(twilio_response))