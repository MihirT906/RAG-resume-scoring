import os
import pandas as pd
from docx import Document
# import textract
import PyPDF2

def read_docx(file_path):
    """Read text from a .docx file."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# def read_doc(file_path):
#     """Read text from a .doc file."""
#     return textract.process(file_path).decode('utf-8')

def read_pdf(file_path):
    """Read text from a .pdf file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in range(len(reader.pages)):
            text.append(reader.pages[page].extract_text())
        return "\n".join(text)

def load_data(docs_file_path):
    """Load resumes and job description from .docx, .doc, and .pdf files."""
    docs_list = pd.DataFrame(columns=['id', 'body'])
    for file_name in os.listdir(docs_file_path):
        if file_name.endswith('.docx'):
            file_path = os.path.join(docs_file_path, file_name)
            resume_text = read_docx(file_path)
        elif file_name.endswith('.doc'):
            file_path = os.path.join(docs_file_path, file_name)
            resume_text = read_docx(file_path)
        elif file_name.endswith('.pdf'):
            file_path = os.path.join(docs_file_path, file_name)
            resume_text = read_pdf(file_path)
        else:
            continue
        
        key = file_name.split()[0]  
        docs_list.loc[len(docs_list)] = {'id': key, 'body': resume_text}

    return docs_list
