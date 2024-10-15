import streamlit as st

def main():
    st.title("Hello World Streamlit App")
    
    st.write("Welcome to my first Streamlit app!")
    
    name = st.text_input("Enter your name")
    if name:
        st.write(f"Hello, {name}!")
    
    st.header("Simple Counter")
    count = st.button("Click me!")
    if "count" not in st.session_state:
        st.session_state.count = 0
    if count:
        st.session_state.count += 1
    st.write(f"Button clicked {st.session_state.count} times")

if __name__ == "__main__":
    main()