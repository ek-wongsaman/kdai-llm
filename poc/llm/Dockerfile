#required installation
#%pip install llama-index
#%pip install llama-index-readers-elasticsearch
#%pip install llama-index-vector-stores-opensearch
#%pip install llama-index-embeddings-ollama
#%pip install ollama
#%pip install nest-asyncio
#%pip install llama-index-embeddings-huggingface
#%pip install fastapi uvicorn

FROM python:3.12.7-slim

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install curl
RUN apt-get update && apt-get install -y curl

# Install ping and other utilities
RUN apt-get update && apt-get install -y iputils-ping

# Install Ollama (modify this part according to the actual Ollama installation instructions)
#RUN curl -O https://ollama.ai/install.sh && bash install.sh

# Continue with your application setup
#for ollama
#WORKDIR /app
#COPY . /app

# Install required Python libraries
RUN pip install --no-cache-dir llama-index \
    llama-index-readers-elasticsearch \
    llama-index-vector-stores-opensearch \
    llama-index-embeddings-ollama \
    ollama \
    llama-index-embeddings-huggingface \
    fastapi \
    uvicorn

RUN pip install --upgrade ollama

#RUN pip install --no-cache-dir curl
#curl is not python package, it cannot be installed via pip.

# Verify Ollama installation
#RUN ollama --version

#RUN ollama pull llama3.2:latest

# Expose the port for the application
EXPOSE 8000

# Run the application using Uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "llm:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "llm-infer-rag:app", "--host", "0.0.0.0", "--port", "8000"]