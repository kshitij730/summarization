import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to summarize text using BART
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.title("PDF Summarizer with BART")

    # Specify the PDF file path
    pdf_file_path = "file.pdf"

    st.write("Using PDF file at path:", pdf_file_path)
    st.text("Original Text:")

    # Extract text from the PDF file
    text_from_pdf = extract_text_from_pdf(pdf_file_path)
    st.write(text_from_pdf)

    # Summarize the extracted text
    summary = summarize_text(text_from_pdf)

    st.text("Summarized Text:")
    st.write(summary)
