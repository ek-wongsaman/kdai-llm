'''
'''
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import nest_asyncio
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

    '''
    '''

    '''
    4. Generate embeddings using the BAAI/bge-m3 model
    In this step, we’ll use the BAAI/bge-m3 model to generate embeddings for our text chunks.
    These embeddings are crucial for semantic search and understanding the content of our documents.
    '''
    #from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embedding_model_name = 'BAAI/bge-m3'
    embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name,max_length=512, device=device)

    embeddings = embedding_model.get_text_embedding("box")
    dim = len(embeddings)

    '''
    5. Setup and Initialize OpenSearch Vector Client
    In this step, we’ll set up and initialize the OpenSearch Vector Client,
    preparing our system to use OpenSearch as a vector database for efficient similarity searches
    in the future.
    '''
    #5.1 Set up OpenSearch connection details
    #this step is mandatory so that the embeding / vector can be stored in the opensearch.

    # http endpoint for your cluster (opensearch required for vector index usage)
    #endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
    endpoint = getenv("OPENSEARCH_ENDPOINT", "http://192.168.50.204:9200") #host machine ip address
    #index to demonstrate the VectorStore impl
    idx = getenv("OPENSEARCH_INDEX", "test_pdf_index")

    #5.2 Configure OpenSearchVectorClient
    # OpensearchVectorClient stores text in this field by default
    text_field = "content_text"
    # OpensearchVectorClient stores embeddings in this field by default
    embedding_field = "embedding"
    # OpensearchVectorClient encapsulates logic for a
    # single opensearch index with vector search enabled with hybrid search pipeline
    client = OpensearchVectorClient(
        endpoint=endpoint,
        index=idx,
        dim=dim,
        embedding_field=embedding_field,
        text_field=text_field,
        search_pipeline="hybrid-search-pipeline",
    )

    '''
    5.3 Initialize OpensearchVectorStore
    We create an instance of OpensearchVectorStore using the configured client.
    This provides an interface for managing the vector store within the Llama Index framework,
    facilitating easy integration with other Llama Index functionalities.
    '''

    # initialize vector store
    vector_store = OpensearchVectorStore(client)

    '''
    6. Create VectorStoreIndex and Store Embeddings
    In this step, we’ll create a VectorStoreIndex using Llama Index and explicitly store
    our document embeddings in OpenSearch. This process enables efficient semantic search and
    retrieval of information from our processed documents.
    '''

    '''
    6.1 Create a StorageContext
    Here, we create a StorageContext using the vector_store (OpenSearch) we initialized in
    the previous step. This StorageContext provides a standardized interface for managing storage
    in Llama Index and connects our index to the OpenSearch vector store.
    '''

    #from llama_index.core import VectorStoreIndex, StorageContext

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    '''
    6.2 Create VectorStoreIndex, Generate and Store Embeddings
    In this crucial step, we create the VectorStoreIndex, generate embeddings for our documents,
    and store them in OpenSearch:
    '''

    index = VectorStoreIndex(
        storage_context=storage_context, embed_model=embedding_model
    )

    '''
    hybrid search pipeline creation will be performed outside of this code.
    vector database suppose to be created and persisted and not initated in every run.
    for the kdai llm final project, it will be initiated in command line using curl command.
    this will call opensearch api to initiate the vector database.
    '''

    '''
    7. Use the index to answer questions about the PDF content
    In this section, we’ll use our created index to answer questions about the content of
    our PDF documents.
    '''

    '''
    7.1 Set up the retriever
    We create a retriever from our index with the following parameters:

    similarity_top_k=3: This specifies that we want to retrieve the top 3 most similar results.

    vector_store_query_mode=VectorStoreQueryMode.HYBRID: This enables hybrid search,
    combining both keyword and semantic search for better results.
    '''

    #from llama_index.core.vector_stores.types import VectorStoreQueryMode
    retriever = index.as_retriever(similarity_top_k=3,vector_store_query_mode=VectorStoreQueryMode.HYBRID)

    '''
    7.2 Define the query
    This is the question we want to ask about our PDF content.

    this part we should use our own question. it is the question regarding legal principle
    from article 291 criminal law.
    '''

    '''
    this question should be input from streamlit app.
    question2 = "นายดำเข้าแย่งมีดจากนายแดง และนายดำถูกแทงเสียชีวิต หากข้อเท็จจริงฟังได้ว่านายดำสดุดเท้าตัวเองแล้วล้มมาโดนมีดแทงจุดสำคัญเป็นเหตุให้ถึงแก่ความตาย นายแดงจะอ้างว่าเป็นการป้องกันตัวโดยชอบด้วยกฎหมาย ตาม มาตรา 68 ได้หรือไม่ เพราะอะไร ให้ยกตัวอย่างคำพิพากษาศาลฎีกาหรือหมายเลขคดีแดงเรื่องนี้ด้วย"
    '''

    question = input_text

    '''
    7.3 Retrieve relevant information
    We use the retriever to find the most relevant information from our indexed documents
    based on the query.
    '''

    prompt = retriever.retrieve(question)

    '''
    7.4 Display the results
    We iterate through the retrieved results, printing both the metadata and the content of each
    retrieved chunk.

    This step demonstrates how we can use our indexed documents to retrieve relevant information
    based on a natural language query. The hybrid search mode allows us to leverage
    both keyword matching and semantic similarity, potentially providing more accurate
    and contextually relevant results.

    By using this approach, we can efficiently answer questions about the content of our PDF documents,
    even if the exact wording doesn’t match what’s in the documents.
    This is particularly useful for creating intelligent document querying systems
    or chatbots that can understand and respond to questions about specific document contents
    '''

    #for r in prompt:
    #    print(r.metadata)
    #    print(r)

    '''
    8. Implement Few-Shot Learning with OpenThaiGPT on Ollama
    8.1 Set up Ollama with Llama3.2.
    This will be done in the command line to ensure model is initiated and setup correctly before
    running inference.
    '''

    '''
    8.2 Create a function to query Llama3.2
    This function takes the user’s question and the retrieved context,
    formats them into a prompt, and sends it to the Llama3.2 model running on Ollama.
    '''

    #answer = answer_question(question)
    answer = query_openthaigpt(question, prompt[0])
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
    response = ollama.generate(model='llama3.2:latest', prompt=formatted_prompt)
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
    return {"original_text": input_data.text, "processed_text": processed_text}

# Run the application using: uvicorn app:app --reload
#first app is python file name, second app is the FastAPI instance name
#running this application uisng uvicorn llm:app -- reload

#----------


