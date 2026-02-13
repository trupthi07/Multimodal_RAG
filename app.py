import streamlit as st
import time
import os
from pypdf import PdfReader

# Corrected import: renamed 'retriver' to 'retriever' to match your file system
from embeddings import get_jina_embeddings
from vision import describe_image
from chunking import chunk_text
from retriever import FAISSRetriever  
from reranker import simple_rerank
from llm import ask_llm

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body { font-family: "Inter", sans-serif; }
    .main-title { font-size: 40px; font-weight: 700; color: #111827; margin-bottom: 0px; }
    .subtitle { font-size: 16px; color: #6b7280; margin-top: 4px; margin-bottom: 25px; }
    .panel { padding: 18px; border-radius: 14px; background-color: #f9fafb; border: 1px solid #e5e7eb; margin-bottom: 18px; }
    .section-header { font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #111827; }
    .stButton button { border-radius: 10px; padding: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>Enterprise Multimodal RAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Retrieval-Augmented Generation over documents and images using Jina Embeddings and Groq Vision</div>",
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "llama-3.2-11b-vision-preview"] # Added a vision model option
    )

    filter_type = st.radio(
        "Retrieval Scope",
        ["all", "text", "image"],
        horizontal=True
    )

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Document Upload</div>", unsafe_allow_html=True)
    txt_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Image Upload</div>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

# Main Processing Logic
if txt_file and groq_key and jina_key:
    with st.spinner("Processing knowledge sources..."):
        # Handle PDF vs Text
        if txt_file.name.endswith(".pdf"):
            reader = PdfReader(txt_file)
            raw_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        else:
            raw_text = txt_file.read().decode("utf-8")

        # Chunking and Metadata
        chunks = chunk_text(raw_text)
        metadata = [{"type": "text"} for _ in chunks]

        # Process Image if exists
        if img_file:
            image_bytes = img_file.read()
            vision_text = describe_image(image_bytes, groq_key)
            if vision_text:
                chunks.append("Image description: " + vision_text)
                metadata.append({"type": "image"})

        # Initialize Retriever
        embeddings = get_jina_embeddings(chunks, jina_key)
        retriever = FAISSRetriever(embeddings, metadata)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Query Interface</div>", unsafe_allow_html=True)
    query = st.text_input("Enter your question", placeholder="Example: What does the uploaded image explain?")
    run = st.button("Run Retrieval and Generate Answer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run and query:
        start = time.time()
        query_emb = get_jina_embeddings([query], jina_key)

        f = None if filter_type == "all" else filter_type
        ids = retriever.search(query_emb, top_k=5, filter_type=f)

        retrieved_docs = [chunks[i] for i in ids]
        reranked = simple_rerank(query, retrieved_docs)

        context = "\n\n".join(reranked[:3])
        answer = ask_llm(context, query, groq_key, model)
        latency = round(time.time() - start, 2)

        st.session_state.history.append((query, answer))

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Answer</div>", unsafe_allow_html=True)
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.metric("Latency (seconds)", latency)

        with st.expander("Retrieved Context"):
            st.text(context)

        with st.expander("Recent Chat History"):
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.divider()
else:
    st.info("Please upload a document and provide API keys in the sidebar to begin.")
