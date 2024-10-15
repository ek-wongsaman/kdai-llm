import streamlit as st
import requests

# Function to call the API
def call_api(text_input):
    try:
        # Example API URL (replace with your actual API endpoint)
        api_url = "http://192.168.50.204:8001/process-text"
        #response = requests.post(api_url, json={"original_text": text_input}, verify=False)
        response = requests.post(api_url, json={"text": text_input}, verify=False)
        
        # Assuming the response is in JSON format
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.title("Streamlit API POC")

# Multiline text input
text_input = st.text_area("Enter your text here:", height=200)

# Submit button
if st.button("Submit"):
    st.write("Calling API...")
    result = call_api(text_input)
    
    # Display a part of the result (assuming the result has a 'processed_text' field)
    if 'error' in result:
        st.error(result['error'])
    else:
        st.text_area("API Response:", result.get('processed_text', 'No data received'), height=100)
    
    # Debug information (prints in the browser)
    st.text(f"Debug: API called with input:\n{text_input}")
else:
    st.write("Waiting for input...")
