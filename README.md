ASK, AI Knowledge Assistant

Document retrieval with traceable evidence.

ASK is an early-stage document intelligence prototype designed to help users find relevant information within PDF documents.

The application uses text extraction, TF-IDF vectorization, and cosine similarity to retrieve passages related to a user’s question and display their source page.

Current Features

* Upload and process a text-based PDF.
* Extract and index document content.
* Search for relevant passages using natural-language questions.
* Display up to three matching passages with page references and similarity scores.
* Show a no-evidence message when no passage meets the retrieval threshold.

Technology Stack

* Python — Application logic
* Streamlit — Interactive web interface
* pypdf — PDF text extraction
* scikit-learn — TF-IDF vectorization and cosine similarity
* NumPy — Result ranking

Getting Started

Clone the repository and install the dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Upload a text-based PDF and enter a question about its contents.

Current Limitations

ASK is currently a retrieval-based prototype, not a generative AI chatbot. It returns passages from uploaded documents rather than generating new answers.

The current implementation supports one PDF at a time. Scanned PDFs without extractable text are not supported, and retrieval quality has not yet been evaluated on a representative document dataset.

Development Roadmap

* [x]	Initial PDF extraction and retrieval engine
* [x]	Streamlit interface
* [x]	Initial synthetic retrieval smoke tests
* [ ]	Evaluate retrieval quality on realistic documents
* [ ]	Improve document chunking and retrieval accuracy
* [ ]	Deploy and validate a publicly accessible application
* [ ]	Explore grounded answer generation with source citations

Project Status

Early-stage engineering prototype, under active development.

Developed by Aylen Alshaikh as part of an ongoing AI engineering and cloud application portfolio.
