# **Lightweight Healthcare Chatbot**

A conversational AI system designed to assist with healthcare-related queries, leveraging a combination of Retrieval-Augmented Generation (RAG) and fine-tuned lightweight models. The chatbot supports both **SMS** and **voice-based interactions**, making it scalable for real-world deployments.

---

## **Features**

- **Retrieval-Augmented Generation (RAG)**:
  - Combines knowledge retrieval and language model generation for accurate responses.
- **Fine-Tuned LLM (DistilGPT-2)**:
  - Custom-trained to handle healthcare-specific queries.
- **Context-Aware Conversations**:
  - Multi-turn dialogue management for better user engagement.
- **SMS and Voice Integration**:
  - Supports Twilio for SMS and Google Dialogflow for voice-based interaction.
- **Scalable Architecture**:
  - Modular design, suitable for local testing and cloud deployment.
- **FastAPI Backend**:
  - Handles user queries, context retrieval, and response generation.

---

## **Getting Started**

### **Prerequisites**

Ensure the following tools and packages are installed:
1. Python 3.8 or above
2. Virtual environment (optional but recommended)
3. Required libraries:
   ```bash
   pip install -r requirements.txt

Includes:

FastAPI
LangChain and LangChain-Community
HuggingFace Transformers
FAISS for vector search
Twilio SDK (for SMS integration)
Dialogflow SDK (for voice integration)

4. Install and configure ngrok for local testing:
  ```ngrok http 8000 ```
5. Installation
1. Clone the repository:
    ```bash 
    git clone https://github.com/yourusername/lightweight-healthcare-chatbot.git
    cd lightweight-healthcare-chatbot
   
2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    venv\Scripts\activate     # For Windows
3. Install dependencies:
    ```bash 
    pip install -r requirements.txt

### ** Data Pipeline ** 
#### Dataset Preparation
Training Dataset: A combination of healthcare FAQs.
Fine-tuning the language model with healthcare-specific instructions and dialogues:
JSON format:
json
```
  {
      "instruction": "What are the symptoms of diabetes?",
      "context": "",
      "response": "Increased thirst, frequent urination, fatigue, blurred vision."
  }
```

#### Embedding Generation
Generate embeddings for your retrieval system:
    
    python app/generate_embeddings.py

#### Fine tuning
Fine Tuning the model for improving the responses:
    
    python train/train.py

#### Running the Application
Local Deployment
Start the FastAPI server:
    ```uvicorn app.app:app --reload```
Access the chatbot interface: Open http://127.0.0.1:8000 in your browser.

#### Test SMS and Voice integrations:

Set up Twilio and Dialogflow credentials in app/config.py.
Live Testing with ngrok
Run ngrok for public access:
  ```bash
  ngrok http 8000
  ```

Update the apiUrl in index.html with the ngrok URL.
Model Training
Fine-tune the language model:
``` 
  python train/train.py 
  ```

Update the fine-tuned model in the retrieval pipeline.
Testing
Run tests for all modules:
pytest app/tests
Integration Details
SMS Integration
Twilio Setup:
Configure your Twilio account and phone number.
Update your Twilio credentials in app/config.py.
Voice Integration
Dialogflow Setup:
Create a Dialogflow agent for voice input handling.
Link your webhook endpoint to the FastAPI backend.
Future Enhancements
Improved Retrieval Efficiency:
Explore faster embedding models like MiniLM.
Knowledge Base Expansion:
Incorporate additional healthcare datasets (e.g., PubMedQA).
Deployment:
Scale the chatbot using Kubernetes and cloud platforms.
Contributing
Feel free to contribute to this project! Open an issue or create a pull request with your suggestions and fixes.
