from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(query: str, context: str):
    model_name = "distilgpt2" #"/home/ubuntu/Desktop/lightweight_chatbot/app/model_checkpoints/llm"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Construct a clear and concise input prompt
    input_text = (
        f"Context: {context}\n"
        f"User Query: {query}\n"
        f"Bot Response:"
    )

    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Explicit attention mask
        max_length=400,
        temperature=0.3,
        top_p=0.9,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(input_text):
        response = response[len(input_text):].strip()

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




