import streamlit as st
import requests
import time

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
INGEST_URL = f"{API_BASE_URL}/ingest-url"
QUERY_URL = f"{API_BASE_URL}/query"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Web RAG Engine", layout="wide")
st.title("🌐 Scalable Web-Aware RAG Engine")

st.info("This app interacts with a FastAPI backend. Ensure the API (`api.py`) and Worker (`worker.py`) are running.")

# --- UI Columns ---
col1, col2 = st.columns(2)

# --- Column 1: Ingestion ---
with col1:
    st.header("1. Ingest Content")
    st.markdown("Add a URL to the knowledge base. The worker will process it in the background.")
    
    with st.form("ingest_form"):
        url_to_ingest = st.text_input(
            "Enter URL", 
            "https://quotes.toscrape.com/scroll"
        )
        submitted = st.form_submit_button("Ingest")

    if submitted and url_to_ingest:
        with st.spinner(f"Sending {url_to_ingest} to ingestion queue..."):
            try:
                response = requests.post(INGEST_URL, json={"url": url_to_ingest})
                
                if response.status_code == 202:
                    data = response.json()
                    st.success(f"**Job Queued!** Message: `{data['message']}` (Job ID: {data['job_id']})")
                    st.info("The background worker will now process this URL.")
                elif response.status_code == 400:
                    st.warning(f"Warning: {response.json()['detail']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.ConnectionError:
                st.error("Connection Failed. Is the FastAPI server running at `{API_BASE_URL}`?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Column 2: Querying ---
with col2:
    st.header("2. Query Knowledge Base")
    st.markdown("Ask a question based on the content you've ingested.")
    
    query = st.text_area("Your Question", "What is RAG?")
    
    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base and generating answer..."):
                try:
                    response = requests.post(QUERY_URL, json={"query": query})
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader("Answer")
                        st.markdown(data['answer'])
                        
                        st.subheader("Sources")
                        if data['sources']:
                            for source in data['sources']:
                                st.write(f"- {source}")
                        else:
                            st.write("No specific sources cited (but answer is from knowledge base).")
                            
                    elif response.status_code == 404:
                         st.error(f"Error: {response.json()['detail']}")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")

                except requests.ConnectionError:
                    st.error(f"Connection Failed. Is the FastAPI server running at `{API_BASE_URL}`?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
