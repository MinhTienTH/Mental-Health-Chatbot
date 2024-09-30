from langchain_community.llms import HuggingFaceHub

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Update HuggingFaceEmbeddings initialization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Update HuggingFaceHub initialization
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.4, "max_length": 1000},
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

# Update chain invocation
def get_response(query):
    result = chain.invoke({"query": query})
    return result

load_dotenv()

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.4, "max_length": 1000}
)

embeddings = HuggingFaceEmbeddings()
vector_db_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="Mental_Health_FAQ.csv", source_column="Questions")
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(vector_db_path)


def create_chain():
    # Load the vector database with allow_dangerous_deserialization set to True
    vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """
    You are an AI assistant that helps people with mental health problems.
    You are given a query and a list of answers.
    You need to select the most appropriate answer based on the query.
    You need to provide the most accurate and helpful answer to the user's question.

    Context: {context}
    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        input_key="query",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain
    

if __name__ == "__main__":
    chain = create_chain()
    while True:
        query = input("Enter a query: ")
        if query == "exit":
            break
        result = chain(query)
        print(result)