import streamlit as st
import requests
import time
import logging

# --- Config ---
st.set_page_config(page_title="Web RAG Engine", layout="wide")
st.title("📚 Web-Aware RAG Engine")
logging.basicConfig(level=logging.INFO)

# --- API URL ---
try:
    # This works on Streamlit Cloud/Render by reading the secret file
    API_URL = st.secrets["FASTAPI_BACKEND_URL"]
except KeyError:
    st.error("FASTAPI_BACKEND_URL not found in Streamlit secrets. Please add it.")
    logging.error("FASTAPI_BACKEND_URL not found in Streamlit secrets.")
    st.stop()
except FileNotFoundError:
    # This is a fallback for local development if secrets.toml is missing
    logging.warning("Streamlit secrets not found, falling back to default local URL.")
    API_URL = "http://127.0.0.1:8000"

# --- Ingestion ---
st.header("1. Ingest a URL")
with st.form("ingest_form"):
    url_to_ingest = st.text_input(
        "Enter a URL to process:", 
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    )
    submitted_ingest = st.form_submit_button("Ingest URL")

if submitted_ingest:
    if not url_to_ingest:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Submitting URL to ingestion queue..."):
            try:
                response = requests.post(
                    f"{API_URL}/ingest-url",
                    json={"url": url_to_ingest}
                )
                
                if response.status_code == 202:
                    st.success(f"**Success!** Message: `{response.json()['message']}` (Status: {response.json()['status']})")
                elif response.status_code == 200:
                     st.info(f"**Info:** Message: `{response.json()['message']}` (Status: {response.json()['status']})")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to backend at {API_URL}.")
            except Exception as e:
                st.error(f"An unknown error occurred: {e}")

# --- Querying ---
st.header("2. Query the Knowledge Base")
with st.form("query_form"):
    query_text = st.text_input("Ask a question based on the ingested content:")
    submitted_query = st.form_submit_button("Get Answer")

if submitted_query:
    if not query_text:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching vectors and generating answer..."):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": query_text}
                )
                end_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Answer found in {end_time - start_time:.2f} seconds.")
                    st.markdown("#### Answer:")
                    st.markdown(data['answer'])
                    
                    st.markdown("#### Sources:")
                    for source_url in data['sources']:
                        st.write(f"- {source_url}")
                elif response.status_code == 404:
                    st.warning("No relevant information was found to answer that question.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to backend at {API_URL}.")
            except Exception as e:
                st.error(f"An unknown error occurred: {e}")

# --- System Health ---
st.sidebar.header("System Health")
if st.sidebar.button("Check Health"):
    with st.sidebar:
        with st.spinner("Pinging services..."):
            try:
                health_res = requests.get(f"{API_URL}/health")
                if health_res.status_code == 200:
                    health_data = health_res.json()
                    st.json(health_data)
                    if all(v == "ok" for v in health_data.values()):
                        st.success("All systems operational.")
                    else:
                        st.warning("One or more services are reporting errors.")
                else:
                    st.error(f"API unresponsive (Code: {health_res.status_code})")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Failed to connect to API at {API_URL}")
            except Exception as e:
                st.error(f"Failed to check health: {e}")
