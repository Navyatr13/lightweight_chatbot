from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(query: str, context: str):
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = f"{context}\n\n{query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(
    input_ids,
    max_length=400,
    no_repeat_ngram_size=3,  # Prevent repeating n-grams of this size
    num_beams=5,            # Beam search for better quality
    early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    

def generate_response_old(query: str, context: str, model_name="gpt-3.5-turbo"):
    """
    Generate a chatbot response based on the query and retrieved context.
    Args:
        query (str): The user query.
        context (str): The retrieved context.
        model_name (str): Name of the language model to use.
    Returns:
        str: Generated response.
    """
    # Initialize the LLM
    llm = ChatOpenAI(model=model_name)
    
    # Combine query and context
    input_message = f"Context: {context}\n\nQuery: {query}"

    # Generate response
    response = llm([HumanMessage(content=input_message)])
    return response.content




