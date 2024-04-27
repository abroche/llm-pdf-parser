import json
from flask import Flask, render_template, request, jsonify
import requests
import logging
import fitz  # PyMuPDF for PDF parsing
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
# from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
import textwrap
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyMuPDFLoader


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



app = Flask(__name__, static_url_path='/static')

# load_dotenv(find_dotenv())
embeddings = OllamaEmbeddings()
db = None
chat_history = []


@app.route('/send_prompt', methods=['POST'])
def send_prompt():
    global chat_history
    prompt = request.form['prompt']
    chat_history.append({"role": "user", "content": prompt})

    output = get_response_from_query(prompt)
    
    return jsonify({"answer": output})
    


def create_db_from_pdf(pdf_path: str) -> FAISS:
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(data)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(query, k=4) -> str:
    global db, chat_history

    if db is None:
        return "No document has been uploaded yet." 

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = Ollama(model="llama2")
    logging.debug("Made the model")

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that that can answer questions based on the chat history and the PDF content
        provided below.
        {history} 
        
        Answer the following question: {input}
        By searching the following document: {context}
        
        Only use the factual information from the document to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = db.as_retriever()
    logging.debug("Made retriever")
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    logging.debug("Made retrieval chain")


    history = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history])
    response = retrieval_chain.invoke({
        "input": query,
        "context": docs_page_content,
        "history": history
    })
    logging.debug("Received response: %s",response)
    logging.debug("Answer from Received response: %s",response["answer"])
    chat_history.append({"role": "assistant", "content": response['answer']})
    answer = response["answer"].replace("\n", "")
    return answer


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global db
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('upload.html', message='No selected file')

        # Check if the file is a PDF
        if file and file.filename.endswith('.pdf'):
            # Save the uploaded PDF file
            pdf_path = f"static/uploads/{file.filename}"
            file.save(pdf_path)

            # Parse the PDF and extract text
            db = create_db_from_pdf(pdf_path)

            return render_template('upload.html', message='File uploaded successfully')

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
