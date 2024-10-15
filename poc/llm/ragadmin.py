# Specify the path to the folder on the external disk
external_disk_path = "/Volumes/SS2305_1MH01/data-src/kdai-llm-final-20241007/pdf_corpus"  # macOS/Linux
# external_disk_path = "D:/pdf_corpus"  # Windows

'''
'''
from llama_index.core import SimpleDirectoryReader
#reader = SimpleDirectoryReader(input_dir="pdf_corpus",recursive=True)
reader = SimpleDirectoryReader(input_dir=external_disk_path,recursive=True)
documents = reader.load_data()

#3.2 Split documents into chunks:
from llama_index.core.node_parser import TokenTextSplitter
splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separator=" ",
)
token_nodes = splitter.get_nodes_from_documents(
    documents, show_progress=True
)

'''
4. Generate embeddings using the BAAI/bge-m3 model
In this step, we’ll use the BAAI/bge-m3 model to generate embeddings for our text chunks.
These embeddings are crucial for semantic search and understanding the content of our documents.
'''
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embedding_model_name = 'BAAI/bge-m3'
embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name,max_length=512, device=device)

embeddings = embedding_model.get_text_embedding("box")
dim = len(embeddings)
print("embedding dimension of example text ===>",dim)

'''
5. Setup and Initialize OpenSearch Vector Client
In this step, we’ll set up and initialize the OpenSearch Vector Client,
preparing our system to use OpenSearch as a vector database for efficient similarity searches
in the future.
'''

#5.1 Set up OpenSearch connection details
#this step is mandatory so that the embeding / vector can be stored in the opensearch.

from os import getenv
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)

# http endpoint for your cluster (opensearch required for vector index usage)
endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
# index to demonstrate the VectorStore impl
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

from llama_index.core import VectorStoreIndex, StorageContext

storage_context = StorageContext.from_defaults(vector_store=vector_store)

'''
6.2 Create VectorStoreIndex, Generate and Store Embeddings
In this crucial step, we create the VectorStoreIndex, generate embeddings for our documents,
and store them in OpenSearch:
'''

index = VectorStoreIndex(
    token_nodes, storage_context=storage_context, embed_model=embedding_model
)