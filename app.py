import re
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ASK | Knowledge Intelligence",
    page_icon="📖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color: #0C111B; color: #F3F5F9;}
.block-container {max-width: 1050px; padding-top: 3rem;}
h1 {font-size: 3rem !important; letter-spacing: -2px;}
div[data-testid="stMetric"] {
    background: #171F2E;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

def normalize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(
        w[:-1] if len(w) > 4 and w.endswith("s")
        and not w.endswith("ss") else w
        for w in words
    )

@st.cache_resource(show_spinner=False)
def build_index(file_bytes):
    from io import BytesIO
    reader = PdfReader(BytesIO(file_bytes))
    chunks = []

    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                chunks.append({
                    "text": sentence,
                    "page": page_number
                })

    if not chunks:
        raise ValueError("No readable text found in this PDF.")

    vectorizer = TfidfVectorizer(
        preprocessor=normalize,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(
        [c["text"] for c in chunks]
    )
    return chunks, vectorizer, matrix

def retrieve(question, chunks, vectorizer, matrix):
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix)[0]
    ranked = np.argsort(scores)[::-1]

    return [
        {
            **chunks[i],
            "score": round(float(scores[i]), 3)
        }
        for i in ranked[:3]
        if scores[i] >= 0.15
    ]

st.caption("ASK / DOCUMENT INTELLIGENCE")
st.title("Knowledge, with evidence.")
st.write(
    "Upload a PDF, ask questions, and explore relevant "
    "information with traceable sources."
)

st.divider()

uploaded = st.file_uploader(
    "Upload your knowledge base",
    type=["pdf"]
)

if uploaded:
    try:
        chunks, vectorizer, matrix = build_index(
            uploaded.getvalue()
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Document", "1 PDF")
        col2.metric("Indexed sections", len(chunks))
        col3.metric("Search method", "TF-IDF")

        st.success("Document indexed successfully.")

        question = st.chat_input(
            "Ask a question about your document..."
        )

        if question:
            with st.chat_message("user"):
                st.write(question)

            results = retrieve(
                question, chunks, vectorizer, matrix
            )

            with st.chat_message("assistant"):
                if not results:
                    st.warning(
                        "I couldn't find supporting evidence "
                        "in this document."
                    )
                else:
                    st.write("**Most relevant evidence:**")
                    st.write(results[0]["text"])
                    st.caption(
                        f"Source: Page {results[0]['page']} "
                        f"| Similarity: {results[0]['score']}"
                    )

                    with st.expander("Additional sources"):
                        for item in results[1:]:
                            st.write(item["text"])
                            st.caption(
                                f"Page {item['page']} "
                                f"| Score: {item['score']}"
                            )

    except Exception as error:
        st.error(f"Unable to process document: {error}")

else:
    st.info("Upload a text-based PDF to begin.")

st.divider()
st.caption(
    "ASK — AI Knowledge Assistant | Engineering Prototype"
)
