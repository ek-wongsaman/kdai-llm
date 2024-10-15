'''
this app is for testing api-endpoint using fastapi. it will be used for troubleshooting purpose.
'''
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data model
class TextInput(BaseModel):
    text: str

# Create a FastAPI instance
app = FastAPI()

# Define the POST endpoint
@app.post("/process-text")
async def process_text(input_data: TextInput):
    # Call the processing function with the input text
    processed_text = 'test ok'
    return {"original_text": input_data.text, "processed_text": processed_text}

# Run the application using: uvicorn app:app --reload
#first app is python file name, second app is the FastAPI instance name
#running this application uisng uvicorn llm:app -- reload

#----------