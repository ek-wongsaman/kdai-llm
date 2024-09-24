# Use the base image
#FROM python:3.9-slim
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy your application files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install torch transformers fastapi uvicorn

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the API using uvicorn
CMD ["uvicorn", "poc-helloworld:app", "--host", "0.0.0.0", "--port", "8000"]
