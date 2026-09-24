import os

import streamlit as st
from dotenv import load_dotenv

from rag import answer_question, chunk_text, load_index, save_index
from sec import download_filing_text, latest_filing

load_dotenv()

st.set_page_config(page_title="SEC Filing AI Research Assistant", page_icon="📄", layout="wide")
st.title("SEC Filing AI Research Assistant")
st.caption("Ask grounded questions about a company's latest SEC 10-K or 10-Q.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in a .env file before indexing or asking questions.")

with st.sidebar:
    st.header("Index a filing")
    cik = st.text_input("Company CIK", value="0000320193", help="Default: Apple Inc.")
    form = st.selectbox("Filing type", ["10-K", "10-Q"])
    if st.button("Fetch & index filing", type="primary"):
        try:
            with st.spinner("Downloading filing from SEC EDGAR..."):
                filing = latest_filing(cik, form)
                text = download_filing_text(filing)
            with st.spinner("Chunking and generating embeddings..."):
                chunks = chunk_text(text)
                count = save_index(chunks, filing)
            st.success(f"Indexed {count} chunks from {filing['company']} {form} ({filing['filing_date']}).")
            st.link_button("View original SEC filing", filing["url"])
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

records = load_index()
if records:
    first = records[0]
    st.info(
        f"Current index: **{first['company']} {first['form']}**, filed {first['filing_date']} "
        f"— {len(records)} searchable chunks."
    )
else:
    st.info("No filing indexed yet. Use the sidebar to fetch one.")

question = st.text_input(
    "Ask a question",
    placeholder="What are the company's major risk factors?",
)

if st.button("Ask", disabled=not question):
    try:
        with st.spinner("Retrieving relevant filing passages and generating an answer..."):
            answer, sources = answer_question(question)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Retrieved sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i} · similarity {source['score']:.3f}"):
                st.write(source["text"])
                st.caption(f"Chunk {source['chunk_index']} · {source['filing_date']}")
                st.link_button("Open SEC filing", source["url"], key=f"source-{i}")
    except Exception as exc:
        st.error(f"Question answering failed: {exc}")
