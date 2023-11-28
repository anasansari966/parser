import openai
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import shutil
import fitz  # PyMuPDF
import docx
import re
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from word2number import w2n
import os

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['SHORTLISTED_FOLDER'] = os.path.join(os.getcwd(), 'shortlisted')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
openai.api_key = os.getenv("OPENAI_API_KEY")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SHORTLISTED_FOLDER'], exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_contact_number_from_text(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email_from_text(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_skills_from_text(text, must_have_skills, optional_skills):
    skills = []
    for skill in must_have_skills + optional_skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
            skills.append(skill)
    return skills

def extract_work_experience(text):
    pattern = r"\b\d+\+?\syears"
    matches = re.findall(pattern, text, re.IGNORECASE)
    years = max([int(re.search(r'\d+', match).group()) for match in matches], default=0)
    return years

def extract_numeric_experience(text):
    match = re.search(r'\b\d+\b', text)
    if match:
        return int(match.group())

    words = re.findall(r'\b\w+\b', text)
    for word in words:
        try:
            return w2n.word_to_num(word)
        except ValueError:
            continue
    return 0

def extract_work_experience_with_langchain(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    query = "work experience in number only"
    docs = docsearch.similarity_search(query)
    work_experience_text = chain.run(input_documents=docs, question=query)

    return extract_numeric_experience(work_experience_text)

def validate_work_experience(years_of_experience, min_exp, max_exp):
    return min_exp <= years_of_experience <= max_exp

def process_file(file_path, must_have_skills, optional_skills, min_exp, max_exp, accuracy_threshold):
    extracted_text = extract_text_from_pdf(file_path) if file_path.endswith('.pdf') else extract_text_from_docx(file_path)
    extracted_contact_number = extract_contact_number_from_text(extracted_text)
    extracted_email = extract_email_from_text(extracted_text)
    extracted_skills = extract_skills_from_text(extracted_text, must_have_skills, optional_skills)

    if file_path.endswith('.pdf'):
        extracted_work_experience = extract_work_experience_with_langchain(file_path)
    else:
        extracted_work_experience = extract_work_experience(extracted_text)

    is_experience_valid = validate_work_experience(extracted_work_experience, min_exp, max_exp)

    matched_must_have_skills = [skill for skill in must_have_skills if skill in extracted_skills]
    matched_optional_skills = [skill for skill in optional_skills if skill in extracted_skills]

    must_have_skills_accuracy = (len(matched_must_have_skills) / len(must_have_skills)) * 100 if must_have_skills else 0
    optional_skills_accuracy = (len(matched_optional_skills) / len(optional_skills)) * 100 if optional_skills else 0

    is_shortlisted = must_have_skills_accuracy >= accuracy_threshold and is_experience_valid

    return{
        "contact_number": extracted_contact_number,
        "email": extracted_email,
        "skills": extracted_skills,
        "work_experience_years": extracted_work_experience,
        "must_have_skills_accuracy": must_have_skills_accuracy,
        "optional_skills_accuracy": optional_skills_accuracy,
        "is_experience_valid": is_experience_valid,
        "is_shortlisted": is_shortlisted,
        "file_path": file_path
    }

def save_shortlisted_resume(file_path, filename):
    shortlisted_path = os.path.join(app.config['SHORTLISTED_FOLDER'], filename)
    if os.path.exists(shortlisted_path):
        os.remove(shortlisted_path)
    shutil.move(file_path, shortlisted_path)

@app.route('/resume_parser/ping')
def ping():
    return "Pong"

@app.route('/resume_parser', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')
        must_have_skills = request.form.get('must_have_skills').split(',')
        optional_skills = request.form.get('optional_skills').split(',')
        min_exp, max_exp = map(int, request.form.get('experience').split('-'))
        accuracy_threshold = int(request.form.get('accuracy'))

        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                processed_data = process_file(file_path, must_have_skills, optional_skills, min_exp, max_exp, accuracy_threshold)

                if processed_data['is_shortlisted']:
                    save_shortlisted_resume(file_path, filename)
                else:
                    os.remove(file_path)

                results.append(processed_data)

        return render_template('result.html', data=results)

    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf', 'docx']

if __name__ == '__main__':
    app.run(debug=True)