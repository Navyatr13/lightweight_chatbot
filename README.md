# Lightweight Healthcare Chatbot with LangChain and DistilGPT-2

This project implements a lightweight and modular chatbot designed for healthcare use cases. It leverages **retrieval-augmented generation (RAG)** with DistilGPT-2 to provide accurate and context-aware responses to user queries. The chatbot is built to run efficiently in resource-constrained environments and integrates SMS and voice-based interactions.

---

## **Features**
- **Healthcare-Specific Conversations:**
  - Fine-tuned DistilGPT-2 for healthcare FAQs using symptom-disease datasets.
- **Retrieval-Augmented Generation (RAG):**
  - Combines context retrieval with lightweight language models for improved response accuracy.
- **SMS and Voice Interaction:**
  - Integrated with Twilio for SMS and Google Speech-to-Text for voice input.
- **Lightweight and Scalable:**
  - Uses smaller models like DistilGPT-2 and MiniLM for faster inference.
- **Modular and Extensible:**
  - Clearly defined modules for training, retrieval, embeddings, and response generation.
- **Deployable with FastAPI:**
  - Exposes RESTful APIs for seamless integration into other systems.

---

## **Quick Start**

### **1. Clone the Repository**
    ```bash
    git clone https://github.com/your-repo/lightweight-chatbot.git
    cd lightweight-chatbot ```

### ** 2. Set Up Dependencies
Install the required Python packages:

    ```bash
    pip install -r requirements.txt

3. Fine-Tune the Model
Fine-tune the chatbot for healthcare-specific tasks using the provided training scripts:

bash
Copy code
python train/train.py
4. Build the Vector Store
Generate embeddings and create a vector store for context retrieval:

bash
Copy code
python app/generate_embeddings.py
5. Run the Application
Start the chatbot server:

bash
Copy code
uvicorn app.app:app --reload
The API will be available at http://127.0.0.1:8000.

