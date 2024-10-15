import streamlit as st
import requests

def main():
    # Page title
    st.title("LLM inference final assignment")

    # Input text field for the question
    question = st.text_area("Question", height=150)

    # Button to submit the request to the FastAPI endpoint
    if st.button("Submit"):
        # Define the FastAPI endpoint URL
        fastapi_url = "http://localhost:8000/infer"  # Replace with your FastAPI endpoint URL

        # Send a POST request to the FastAPI endpoint with the question as parameter
        try:
            response = requests.post(fastapi_url, json={"question": question})

            # Display the result from the API call
            if response.status_code == 200:
                result = response.json().get("answer", "No answer found")
                st.write("Result: ", result)
            else:
                st.write("Error: Failed to retrieve the result from the API.")

        except Exception as e:
            st.write(f"Error: {e}")

if __name__ == "__main__":
    main()


