import pandas as pd
from flask import Flask, request, jsonify

# --- CORRECTED IMPORT SECTION ---
from langchain_core.documents import Document  # <-- This was moved
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  # <-- This was moved
from langchain_core.runnables import RunnablePassthrough  # <-- This was moved
from langchain_core.output_parsers import StrOutputParser  # <-- This was moved

# --- END OF IMPORTS ---

# --- 1. Create Flask App ---
app = Flask(__name__)

# --- 2. Load and Prep Data (The "RAG" part) ---
print("Loading data and building FAISS index...")
try:
    # Load the dataset
    df = pd.read_csv('car_sales.csv')

    # Convert each row into a text "document"
    documents = []
    for _, row in df.iterrows():
        text = f"In {row['Year']}, the {row['Brand']} {row['Model']} sold {row['Sales']} units. Notes: {row['Notes']}"
        doc = Document(page_content=text)
        documents.append(doc)

    # Load an embedding model (runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Create the "Retriever"
    retriever = vector_store.as_retriever()

    print("✅ Data loaded and FAISS index built successfully.")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# --- 3. Setup LangChain (The "Generation" part) ---
# Define the prompt template
template = """
You are an expert assistant for CARIAD, answering questions about car sales data.
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know. Do not try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer: 
"""
prompt = ChatPromptTemplate.from_template(template)

# Use a local LLM via llama3
llm = ChatOllama(model="llama3")  # Using llama3 for now as we discussed

# Create the RAG chain
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print("🤖 RAG chain created. Flask API is ready to serve.")


# --- 4. Define the Flask API Endpoint ---
@app.route('/query', methods=['POST'])
def query_assistant():
    """
    This is the Flask API endpoint.
    It expects a JSON request with a "question" key.
    e.g., {"question": "How many sales did the Audi e-tron have?"}
    """
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request"}), 400

        question = data['question']

        # Run the RAG chain
        answer = rag_chain.invoke(question)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)