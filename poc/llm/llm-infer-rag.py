'''
'''
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

import datetime

from os import getenv

#4. Generate embeddings using the BAAI/bge-m3 model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#5.1 Set up OpenSearch connection details
#this step is mandatory so that the embeding / vector can be stored in the opensearch.
from os import getenv
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)

from llama_index.core import VectorStoreIndex, StorageContext

from llama_index.core.vector_stores.types import VectorStoreQueryMode


from fastapi import FastAPI
from pydantic import BaseModel

import ollama
from ollama import Client

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class TextInput(BaseModel):
    text: str

# Define the function to process the text
def process_input_text(input_text: str) -> str:

    # Check if CUDA is available for GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(datetime.datetime.now())

    embedding_model_name = 'BAAI/bge-m3'
    embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name,max_length=512, device=device)

    print("DEBUG:huggingface embedding initialized")
    print(datetime.datetime.now())
    embeddings = embedding_model.get_text_embedding("box")
    print("DEBUG: embedding call successful")
    print(datetime.datetime.now())
    dim = len(embeddings)

    endpoint = getenv("OPENSEARCH_ENDPOINT", "http://192.168.50.91:9200") #host machine ip address
    #index to demonstrate the VectorStore impl
    idx = getenv("OPENSEARCH_INDEX", "test_pdf_index")

    text_field = "content_text"

    embedding_field = "embedding"

    client = OpensearchVectorClient(
        endpoint=endpoint,
        index=idx,
        dim=dim,
        embedding_field=embedding_field,
        text_field=text_field,
        search_pipeline="hybrid-search-pipeline",
    )

    print("DEBUG: opensearch client initialized")
    print(datetime.datetime.now())

    # initialize vector store
    vector_store = OpensearchVectorStore(client)

    print("DEBUG: vector store initialized")
    print(datetime.datetime.now())

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize the VectorStoreIndex with the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,  # Pass the vector store you created earlier
        storage_context=storage_context,
        embed_model=embedding_model
    )
    print("DEBUG: vector store index initialized")
    print(datetime.datetime.now())

    retriever = index.as_retriever(similarity_top_k=3,vector_store_query_mode=VectorStoreQueryMode.HYBRID)
    print("DEBUG:retriever initialized")
    print(datetime.datetime.now())

    question = input_text


    prompt = retriever.retrieve(question)
    print("DEBUG: retrieval successful")
    print(datetime.datetime.now())


    answer = query_openthaigpt(question, prompt[0])
    print("DEBUG: inference successful")
    print(datetime.datetime.now())
    return answer


def query_openthaigpt(question, context):
    formatted_prompt = f'''# Few-shot examples
Example 1:
Context: Mr. A killed a person and claimed it was justifiable self-defense under Section 68 of the Criminal Code.
If the facts indicate that there was a struggle for a firearm and the gun accidentally discharged.
Question: How would the court rule? Please provide an example Supreme Court decision case number or the Red case number (หมายเลขคดีแดง)
Answer: The court would likely determine that the defendant's claim of self-defense is not tenable. Justifiable self-defense under Section 68 requires intentional action. Given that the facts indicate a struggle for the firearm and an accidental discharge, this constitutes a negligent act resulting in death, not an intentional act. Therefore, it does not meet the criteria for justifiable self-defense under the law.
The court would likely rule that this case falls under negligent homicide rather than justifiable self-defense,
as the element of intent, which is crucial for the self-defense claim,
is absent in the scenario where the firearm discharged accidentally during a struggle.
The example Supreme Court decision is Case No. 1597/2562, Red Case No. อ632/2560.

# RAG component
Retrieved information:
{context}

# Actual question
Question: {question}

Please answer the question with Thai language using the information from the examples and the retrieved information above. Focus on troubleshooting end-user devices in FTTX projects. Provide a clear, step-by-step answer, prioritized in order of importance. Include a brief explanation for each step. If the provided information is insufficient to answer the question completely, state "The available information is not sufficient to fully answer this question" and suggest general troubleshooting steps.

Answer:'''

    #print(formatted_prompt)
    #response = ollama.generate(model='openthaigpt:latest', prompt=formatted_prompt)
    #response = ollama.generate(model='llama3.2:latest', prompt=formatted_prompt)   192.168.50.204:
    #response = ollama.generate(model='llama3.2:latest', prompt=formatted_prompt, server_url="http://localhost:11434")
    #ollama_endpoint = getenv("OLLAMA_ENDPOINT", "http://192.168.50.91:11434") #host machine ip address
    ollama_endpoint = getenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434") #hanother approach to refer to host machine ip address
    #use env allow to change ip after deployment
    #ollama.set_host(ollama_endpoint)
    #response = ollama.generate(model='llama3.2:latest', prompt=formatted_prompt, server_url=ollama_endpoint)
    #response = ollama.generate(model='llama3.2:latest', prompt=formatted_prompt)

    client = Client(host=ollama_endpoint)
    response = client.generate(model='llama3.2:latest', prompt=formatted_prompt)

    print("DEBUG: ollama call successful")
    print(datetime.datetime.now())
    return response['response']

'''
in-context learning
8.3 Integrate OpenThaiGPT with our retriever
Now, let’s modify our question-answering process to use OpenThaiGPT:

This function retrieves relevant information using our previously set up retriever,
combines the retrieved text into a context, and then uses OpenThaiGPT to generate an answer
based on this context.
'''

'''
def answer_question(question):   
    # Query OpenThaiGPT
    answer = query_openthaigpt(question, prompt[0])
    return answer
'''

#inference result here will be displayed in streamlit app.

# Define the POST endpoint
@app.post("/process-text")
async def process_text(input_data: TextInput):
    # Call the processing function with the input text
    processed_text = process_input_text(input_data.text)

    print("DEBUG: api processing successful")
    print(datetime.datetime.now())
    return {"original_text": input_data.text, "processed_text": processed_text}

# Run the application using: uvicorn app:app --reload
#first app is python file name, second app is the FastAPI instance name
#running this application uisng uvicorn llm:app -- reload

#----------


