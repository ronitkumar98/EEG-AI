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
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product RAG API",
              description="Retrieval Augmented Generation API for Product Information",
              version="1.0.0")


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
redis_host = os.getenv("REDIS_HOST")
redis_password = os.getenv("REDIS_PASS")
redis_url = f"redis://:{redis_password}@{redis_host}:14266"
MAX_HISTORY_LENGTH = 5  # Number of recent conversation pairs to maintain

# # Initialize Redis client
# redis_client = redis.Redis(
#     host=redis_host,
#     port=14266,
#     password=redis_password,
# )

# Dependency to ensure Redis is initialized


class LimitedRedisChatMessageHistory(RedisChatMessageHistory):
    """Custom chat history class that maintains only recent messages"""

    def __init__(self, session_id: str, redis_client: redis.Redis, max_history: int = MAX_HISTORY_LENGTH):
        super().__init__(session_id=session_id, redis_client=redis_client)
        self.max_history = max_history

    def add_message(self, message):
        """Add a message and trim history if needed"""
        super().add_message(message)
        messages = self.messages

        # If we have more than max_history pairs of messages, trim the oldest ones
        if len(messages) > (self.max_history * 2):
            # Keep only the most recent max_history pairs
            messages = messages[-(self.max_history * 2):]
            # Clear and update Redis with trimmed history
            self.clear()
            for msg in messages:
                super().add_message(msg)


redis_client = None
redis_instance = None


async def get_redis_client():
    if redis_client is None:
        raise HTTPException(
            status_code=503, detail="Redis client not initialized")
    return redis_client


@app.on_event("startup")
async def startup_event():
    global redis_client, redis_instance
    try:
        # Initialize Redis client
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=14266,
            password=os.getenv("REDIS_PASS"),
            decode_responses=True
        )
        # Test Redis connection
        redis_client.ping()

        # Initialize redis_instance (your vector store)
        redis_instance = init_redis_store()

        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "ok", "message": "Product Information Retrieval System is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        return {
            "status": "healthy",
            "services": {
                "redis": "connected",
                "api": "running"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


# Initialize OpenAI components
embed_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

# Define templates
question_template = """You are an expert in summarizing questions.
                    Your goal is to reduce a question to its simplest form while retaining the semantic meaning.
                    Try to be as deterministic as possible
                    Below is the question:
                    {question}
                    Output will be a semantically similar question that will be used to query an existing database."""

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

# Initialize the chain with memory
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: LimitedRedisChatMessageHistory(
        session_id=session_id,
        redis_client=redis_client,
        max_history=MAX_HISTORY_LENGTH
    ),
    input_messages_key="question",
    history_messages_key="history"
)


# Initialize Redis vector store

def init_redis_store():
    try:
        # Attempt to connect to existing index
        return Redis.from_existing_index(
            embedding=embed_model,
            index_name="product_index",
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
    except Exception as e:
        print(f"Existing index not found: {e}")
        print("Creating new Redis index...")
        # Get and prepare data
        data = get_data()
        df = prepare_data(data)
        corpus = create_corpus(df)
        summaries = [create_prod_summary(text) for text in corpus]

        # Create new index
        return Redis.from_texts(
            texts=summaries,
            embedding=embed_model,
            index_name="product_index",
            redis_url=redis_url,
            metadata=[{"id": i} for i in range(len(summaries))],
        )


def get_data() -> List[Dict]:
    """Fetch product data from the API"""
    url = "https://eeg-backend-hfehdmd4hxfagsgu.canadacentral-01.azurewebsites.net/api/users/product"
    response = requests.get(url)
    return response.json()


def prepare_data(data: List[Dict]) -> pd.DataFrame:
    """Clean and prepare the product data"""
    df = pd.DataFrame(data)
    df.fillna("Unknown", inplace=True)
    df["chemicalProperties"] = df["chemicalProperties"].apply(
        lambda x: "Unknown" if len(x) == 0 else x
    )
    return df


def create_corpus(df: pd.DataFrame) -> List[str]:
    """Create a text corpus from the DataFrame"""
    corpus = []
    for i in range(df.shape[0]):
        text = " ".join(f"{col}: {str(df[col][i])}" for col in df.columns)
        corpus.append(text)
    return corpus


def create_prod_summary(text: str) -> str:
    """Create a product summary using ChatGPT"""
    message = f"Here is a product data {text}. Your job is to create a listing of the entire product. Mention all the features and details present in the data."
    return llm.invoke(message).content


def retrieve_docs(question: str) -> str:
    """Retrieve relevant documents for a given question"""
    modified_question = llm.invoke(
        question_template.format(question=question)).content
    redis_result = redis_instance.similarity_search(
        query=modified_question, k=5)
    return "\n".join(res.page_content for res in redis_result)


class AnswerResponse(BaseModel):
    answer: str


@app.get("/api/chat", response_model=AnswerResponse)
@app.post("/api/chat", response_model=AnswerResponse)
async def chat_endpoint(question: str = Query(...)):
    """Handle chat endpoint"""
    session_id = "rag_session_2"
    try:
        context = retrieve_docs(question)
        answer = chain_with_history.invoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
        return AnswerResponse(answer=answer.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)