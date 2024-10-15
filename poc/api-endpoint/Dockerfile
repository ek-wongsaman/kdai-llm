
FROM python:3.12.7-slim

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install curl
#use to test connection in docker for troubleshooting purpose.
RUN apt-get update && apt-get install -y curl

# Install required Python libraries
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn

#RUN pip install --no-cache-dir curl
#curl is not python package, it cannot be installed via pip.

# Expose the port for the application
EXPOSE 8001

# Run the application using Uvicorn
CMD ["uvicorn", "simple-endpoint:app", "--host", "0.0.0.0", "--port", "8001"]