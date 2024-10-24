"""
Product Information Retrieval System with Memory
This script implements a RAG (Retrieval Augmented Generation) system that:
1. Fetches product data from an API
2. Creates vector embeddings stored in Redis
3. Implements a conversation system with memory
4. Retrieves and answers product-related queries
"""

from langchain_redis import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Redis
import redis
import requests
import pandas as pd
import dotenv
from typing import List, Dict
import json
import os
import warnings
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# Standard library imports

# Third-party imports

# Load environment variables
dotenv.load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
redis_host = os.getenv("REDIS_HOST")
redis_password = os.getenv("REDIS_PASS")
redis_url = f"redis://:{redis_password}@{redis_host}:14266"

# Initialize Redis client
redis_client = redis.Redis(
    host=redis_host,
    port=14266,
    password=redis_password,
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

# Define templates for question processing and responses
question_template = """You are an expert in summarizing questions.
                    Your goal is to reduce a question to its simplest form while retaining the semantic meaning.
                    Try to be as deterministic as possible
                    Below is the question:
                    {question}
                    Output will be a semantically similar question that will be used to query an existing database."""

# Define the main conversation prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in answering questions about products.
                 Answer based on the retrieved product data below:
                 {context}

                 For greetings like "Hi" or "Hello", respond politely.
                 If multiple products are relevant, list all of them with the necessary information only.
                 Compare products based on their features and details if the user asks.
                 If you're not sure about something, say so."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def get_data() -> List[Dict]:
    """
    Fetch product data from the API
    Returns:
        List[Dict]: List of product dictionaries
    """
    url = "https://eeg-backend-hfehdmd4hxfagsgu.canadacentral-01.azurewebsites.net/api/users/product"
    response = requests.get(url)
    return response.json()


def prepare_data(data: List[Dict]) -> pd.DataFrame:
    """
    Clean and prepare the product data
    Args:
        data (List[Dict]): Raw product data
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame
    """
    df = pd.DataFrame(data)
    df.fillna("Unknown", inplace=True)
    df["chemicalProperties"] = df["chemicalProperties"].apply(
        lambda x: "Unknown" if len(x) == 0 else x
    )
    return df


def create_corpus(df: pd.DataFrame) -> List[str]:
    """
    Create a text corpus from the DataFrame
    Args:
        df (pd.DataFrame): Product DataFrame
    Returns:
        List[str]: List of product descriptions
    """
    corpus = []
    for i in range(df.shape[0]):
        text = " ".join(f"{col}: {str(df[col][i])}" for col in df.columns)
        corpus.append(text)
    return corpus


def create_prod_summary(text: str) -> str:
    """
    Create a product summary using ChatGPT
    Args:
        text (str): Product description
    Returns:
        str: Generated product summary
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY,
    )
    message = f"Here is a product data {text}. Your job is to create a listing of the entire product. Mention all the features and details present in the data."
    return llm.invoke(message).content


def retrieve_docs(question: str) -> str:
    """
    Retrieve relevant documents for a given question
    Args:
        question (str): User question
    Returns:
        str: Concatenated relevant documents
    """
    modified_question = llm.invoke(
        question_template.format(question=question)).content
    redis_result = redis_instance.similarity_search(
        query=modified_question, k=5)
    return "\n".join(res.page_content for res in redis_result)


def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get Redis chat history for a session
    Args:
        session_id (str): Unique session identifier
    Returns:
        BaseChatMessageHistory: Chat history instance
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        redis_client=redis_client,
    )


def chat_with_memory(question: str, session_id: str) -> str:
    """
    Process a chat message with memory
    Args:
        question (str): User question
        session_id (str): Session identifier
    Returns:
        str: Generated response
    """
    context = retrieve_docs(question)
    response = chain_with_history.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content


# Create FastAPI app
app = FastAPI()

# Define request model


# class QuestionRequest(BaseModel):
#     question: str
#     session_id: str

# Define response model


class AnswerResponse(BaseModel):
    answer: str


@app.post("/api/chat", response_model=AnswerResponse)
# async def chat_endpoint(request: QuestionRequest):
async def chat_endpoint(question: str = Query(...), session_id: str = Query(...)):
    try:
        answer = chat_with_memory(question, session_id)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Initialize OpenAI embeddings
    embed_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY,
    )

    # Get and prepare data
    data = get_data()
    df = prepare_data(data)
    corpus = create_corpus(df)
    index_name = "product_index"

    try:
        # Attempt to connect to existing index
        redis_instance = Redis.from_existing_index(
            embedding=embed_model,
            index_name=index_name,
            redis_url=redis_url,
            schema={
                "summary": "TEXT",
                "id": "NUMERIC",
                "embedding": {
                    "TYPE": "FLOAT32",
                    "DIM": 768,
                    "DISTANCE_METRIC": "COSINE"
                }
            }
        )
        print("Connected to existing Redis index.")
    except Exception as e:
        print(f"Existing index not found: {e}")
        print("Creating new Redis index...")
        # Create summaries and setup Redis
        summaries = [create_prod_summary(text) for text in corpus]
        # Create new index if not found
        redis_instance = Redis.from_texts(
            texts=summaries,
            embedding=embed_model,
            index_name=index_name,
            redis_url=redis_url,
            metadata=[{"id": i} for i in range(len(summaries))],
        )
        print("New Redis index created successfully.")

    # Create the chain with memory
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_redis_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
