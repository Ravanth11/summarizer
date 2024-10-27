from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline


# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)










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
    txt_path = r'demofile3.txt'
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

    # Language codes for the supported languages
    language_codes = {
        "english": "en_XX",
        "hindi": "hi_IN",
        "tamil": "ta_IN",
        "malayalam": "ml_IN"
    }

    # Load the mBART model and tokenizer for translation
    translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    # Get user input for the source language and the text
    # source_language = input("Enter the source language (english, hindi, tamil, malayalam): ").strip().lower()
    source_language = 'english'
    # text_to_translate = input("Enter the text you want to translate: ").strip()
    text_to_translate = answer
    
    if source_language in language_codes:
        # Set the source language based on user input
        translation_tokenizer.src_lang = language_codes[source_language]
        # Tokenize the input text
        encoded_input = translation_tokenizer(text_to_translate, return_tensors="pt")

        # Generate the translation to English
        generated_tokens_en = translation_model.generate(
            **encoded_input,
            forced_bos_token_id=translation_tokenizer.lang_code_to_id["en_XX"],
            max_length=100,
            num_beams=5,
            repetition_penalty=2.5
        )

        # Decode the translated text to English
        translated_text_en = translation_tokenizer.batch_decode(generated_tokens_en, skip_special_tokens=True)[0]
        print("Translated text (to English):", translated_text_en)

        # Perform sentiment analysis on the translated text
        # sentiment_result = sentiment_analysis(translated_text_en)
        # print("Sentiment Analysis Result:", sentiment_result)

        # Ask the user for the target language
        # target_language = input("Enter the target language (hindi, tamil, malayalam): ").strip().lower()
        global target_language
        target_language = 'hindi'

        # Check if the target language is supported
        if target_language in language_codes:
            # Generate the translation to the specified target language
            generated_tokens_target = translation_model.generate(
                **encoded_input,
                forced_bos_token_id=translation_tokenizer.lang_code_to_id[language_codes[target_language]],
                max_length=100,
                num_beams=5,
                repetition_penalty=2.5
            )

            # Decode the translated text to the target language
            global translated_text_target
            translated_text_target = translation_tokenizer.batch_decode(generated_tokens_target, skip_special_tokens=True)[0]
            print(f"Translated text (to {target_language}):", translated_text_target)
        else:
            print(f"Target language '{target_language}' is not supported.")
    else:
        print(f"Source language '{source_language}' is not supported.")
    # Render the summary_display.html template with the generated answer and summary
    if target_language == 'english':
        return render(request, 'summary_display.html', {'answer': answer, 'summary': summary})
    else:
        return render(request, 'summary_display.html', {'answer': answer, 'summary': translated_text_target})