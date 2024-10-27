
#%%
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)










from django.shortcuts import render

import requests
from django.http import HttpResponse

import requests
import requests
import feedparser
from bs4 import BeautifulSoup
from fpdf import FPDF



def index(request):
    return render(request,'index.html')


def url(request):
    # Fetching the content from the URL
    res = requests.get("https://www.geeksforgeeks.org/introduction-of-system-call")
    soup = BeautifulSoup(res.content, "html.parser")
    content = soup.get_text()

    # Writing the content to a file with UTF-8 encoding
    with open("demofile3.txt", "w", encoding="utf-8") as f:
        for line in content.splitlines():
            f.write(line + "\n")

    # Return a response indicating success
    return render(request, 'content_display.html', {'content': content})



def summary(request):
    # Path to the local text file
    txt_path = r'C:\Users\Asus\Desktop\NLP\url\demofile3.txt'

    # Reading text content from the file with UTF-8 encoding
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        return HttpResponse(f"Error reading file: {e}", status=500)

    # Splitting text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)

    # Google API setup
    google_api_key = 'AIzaSyARn_PcqweM5MXHxYaIWGQcf-BDJMP1bDw'
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    # Creating and saving FAISS vector store for efficient similarity search
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    # Loading BART model and tokenizer for text generation and summarization
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    # Prompt template for question answering
    prompt_template = """
    Answer the question in a detailed way and include all the related details. If the answer is not available
    in the provided context, say, 'Answer is not available'. Avoid generating random responses.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """

    # Initialize Google Generative AI model
    qa_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(qa_model, chain_type="stuff", prompt=prompt)

    # Getting the question from the user input (request parameter)
    question = request.GET.get('question', 'What is a system call?')

    # Loading the FAISS index for searching relevant documents
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(question)
    except Exception as e:
        return HttpResponse(f"Error loading FAISS index: {e}", status=500)

    if not docs:
        return HttpResponse("No relevant documents found.")

    # Get the response from the QA chain
    response = chain.invoke({"input_documents": docs, "question": question})
    answer = response.get("output_text", "No answer generated.")

    # If the answer isn't found in the context, generate a summary
    summary = ""
    if "answer is not available" in answer.lower():
        concatenated_text = " ".join([doc.page_content for doc in docs])
        inputs = tokenizer(concatenated_text, max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        answer += "\n\nGenerated Summary/Insight: " + summary

    # Render the summary_display.html template with the generated answer and summary
    return render(request, 'summary_display.html', {'answer': answer, 'summary': summary})
