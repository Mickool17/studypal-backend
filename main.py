from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import requests
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List
import json
import re
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from docx import Document
import tempfile
import shutil
import imghdr  # For image format detection

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class QuestionRequest(BaseModel):
    text: str
    question_type: str = "multiple_choice"
    num_questions: int = 5

class AnswerRequest(BaseModel):
    question: str
    user_answer: str
    correct_answer: str
    question_type: str

class GradeRequest(BaseModel):
    answers: List[AnswerRequest]

# Temporary directory for storing images
TEMP_DIR = tempfile.mkdtemp()

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Study Companion Backend"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.lower().split('.')[-1]
        images = []
        
        if file_extension == 'pdf':
            with pdfplumber.open(file.file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    # Extract images
                    for img in page.images:
                        try:
                            img_data = img['stream'].get_data()
                            # Validate image format
                            img_format = imghdr.what(None, h=img_data)
                            if img_format not in ['png', 'jpeg', 'bmp', 'gif']:
                                print(f"Skipping unsupported image format: {img_format}")
                                continue
                            img_io = BytesIO(img_data)
                            img_pil = Image.open(img_io)
                            img_buffer = BytesIO()
                            img_pil.save(img_buffer, format="PNG")
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            images.append({
                                'figure': f"Figure {len(images) + 1}",
                                'image': img_base64
                            })
                        except UnidentifiedImageError as e:
                            print(f"Error processing PDF image: {str(e)}")
                            continue
                        except Exception as e:
                            print(f"Unexpected error processing PDF image: {str(e)}")
                            continue
        elif file_extension == 'docx':
            temp_file = os.path.join(TEMP_DIR, file.filename)
            with open(temp_file, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            doc = Document(temp_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            # Extract images
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        img_data = rel.target_part.blob
                        # Validate image format
                        img_format = imghdr.what(None, h=img_data)
                        if img_format not in ['png', 'jpeg', 'bmp', 'gif']:
                            print(f"Skipping unsupported image format: {img_format}")
                            continue
                        img_io = BytesIO(img_data)
                        img_pil = Image.open(img_io)
                        img_buffer = BytesIO()
                        img_pil.save(img_buffer, format="PNG")
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        images.append({
                            'figure': f"Figure {len(images) + 1}",
                            'image': img_base64
                        })
                    except UnidentifiedImageError as e:
                        print(f"Error processing DOCX image: {str(e)}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error processing DOCX image: {str(e)}")
                        continue
            os.remove(temp_file)
        elif file_extension == 'txt':
            text = await file.read()
            text = text.decode('utf-8')
        else:
            return {"error": f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .docx"}
        
        print(f"Extracted text length: {len(text)}")
        print(f"Extracted {len(images)} images")
        return {"text": text, "images": images}
    except Exception as e:
        print(f"Error in extract_text: {str(e)}")
        return {"error": str(e), "text": "", "images": []}

@app.post("/generate-questions")
async def generate_questions(request: QuestionRequest):
    try:
        print(f"Received request: {request}")
        num_questions = max(1, min(request.num_questions, 75))
        if request.question_type == "multiple_choice":
            prompt = (
                f"Generate exactly {num_questions} multiple-choice questions based on the following content:\n\n{request.text}\n\n"
                f"Provide 4 answer options per question, with the correct answer marked with **, e.g., c) **Correct Option**. "
                f"Ensure each question is numbered (e.g., Q1, Q2) and formatted clearly with a period after the question number (e.g., Q1.). "
                f"If the content references figures (e.g., Figure 1), include a 'figure' field in the response to indicate the figure number. "
                f"If the content is insufficient, generate relevant questions based on the topic to reach exactly {num_questions} questions."
            )
        else:  # theoretical
            prompt = (
                f"Generate exactly {num_questions} open-ended theoretical questions based on the following content:\n\n{request.text}\n\n"
                f"For each question, provide a sample answer prefixed with 'Sample Answer:'. "
                f"Format each question as: '**Q#: Question text**' and the sample answer as '**Sample Answer: Answer text**'. "
                f"Ensure each question is numbered (e.g., Q1, Q2). "
                f"If the content references figures (e.g., Figure 1), include a 'figure' field in the response to indicate the figure number. "
                f"If the content is insufficient, generate relevant questions based on the topic to reach exactly {num_questions} questions."
            )
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 8000
            }
        }
        response = requests.post(GEMINI_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        questions = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"Gemini raw response length: {len(questions)} characters")
        print(f"Gemini raw response: {questions[:500]}...")

        # Parse questions
        parsed_questions = []
        if request.question_type == "multiple_choice":
            question_matches = re.findall(r'(Q\d+\.\s+.*?)(?=Q\d+\.|$)', questions, re.DOTALL)
            question_count = len(question_matches)
            print(f"Parsed {question_count} multiple-choice questions")
            for q in question_matches:
                lines = q.strip().split('\n')
                question_text = lines[0].replace('Q', '', 1).lstrip('0123456789. ').replace('""', '').strip()
                options = [line.strip().replace('**', '') for line in lines[1:5] if line.strip()]
                correct_option = next((opt for opt in options if '**' in opt), options[0])
                figure_match = re.search(r'Figure\s+(\d+)', question_text)
                figure = figure_match.group(1) if figure_match else None
                question_data = {
                    'text': question_text,
                    'type': 'multiple_choice',
                    'options': options,
                    'correct_answer': correct_option.replace('**', '').strip()
                }
                if figure:
                    question_data['figure'] = f"Figure {figure}"
                parsed_questions.append(question_data)
        else:  # theoretical
            question_matches = re.findall(r'\*\*Q\d+:.*?\*\*\n.*?\n\*\*Sample Answer:.*?(?=\*\*Q\d+:|$)', questions, re.DOTALL)
            question_count = len(question_matches)
            print(f"Parsed {question_count} theoretical questions")
            for q in question_matches:
                lines = q.strip().split('\n')
                question_text = lines[0].replace('**Q', '').lstrip('0123456789: ').replace('**', '').replace('""', '').strip()
                sample_answer = lines[2].replace('**Sample Answer:', '').replace('""', '').strip()
                figure_match = re.search(r'Figure\s+(\d+)', question_text)
                figure = figure_match.group(1) if figure_match else None
                question_data = {
                    'text': question_text,
                    'type': 'theoretical',
                    'correct_answer': sample_answer
                }
                if figure:
                    question_data['figure'] = f"Figure {figure}"
                parsed_questions.append(question_data)

        print(f"Generated {question_count} questions out of {num_questions} requested")
        if question_count < num_questions:
            print(f"Warning: Generated fewer questions ({question_count}) than requested ({num_questions})")

        return {"questions": parsed_questions}
    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        return {"error": str(e)}

@app.post("/grade-answer")
async def grade_answer(request: GradeRequest):
    try:
        results = []
        score = 0.0
        total = len(request.answers)
        print(f"Grading {total} answers")

        for i, answer in enumerate(request.answers):
            print(f"Processing answer {i+1}: {answer.question}")
            partial_score = 0.0
            feedback = ""
            if answer.question_type == "multiple_choice":
                is_correct = answer.user_answer.strip() == answer.correct_answer.strip()
                partial_score = 1.0 if is_correct else 0.0
                feedback = "Correct answer!" if is_correct else "Incorrect, please review the material."
            else:  # theoretical
                if not answer.user_answer.strip():
                    partial_score = 0.0
                    feedback = "No answer provided."
                    is_correct = False
                else:
                    prompt = (
                        f"Evaluate the following user answer for correctness based on this question and expected answer. "
                        f"Return a JSON object wrapped in triple backticks (```json\n{{'partial_score': number, 'feedback': 'string'}}```\n), "
                        f"where 'partial_score' is a percentage (0 to 100) reflecting the answer's quality (e.g., completeness, accuracy), "
                        f"and 'feedback' explains the score, including what was correct and what was missing.\n"
                        f"Question: {answer.question}\n"
                        f"Expected Answer: {answer.correct_answer}\n"
                        f"User Answer: {answer.user_answer}"
                    )
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
                    }
                    response = requests.post(GEMINI_API_URL, json=data, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    grading_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                    print(f"Raw Gemini grading response: {grading_text}")
                    try:
                        if grading_text.startswith('```json'):
                            grading_text = grading_text.split('```json\n')[1].split('\n```')[0]
                        grading_data = json.loads(grading_text)
                        partial_score = min(max(float(grading_data.get("partial_score", 0)), 0), 100) / 100.0
                        feedback = grading_data.get("feedback", "No feedback provided")
                    except (json.JSONDecodeError, IndexError, ValueError) as e:
                        print(f"JSON decode error for answer {i+1}: {str(e)} for response: {grading_text}")
                        partial_score = 0.0
                        feedback = f"Unable to evaluate answer: {grading_text}"
            
            score += partial_score
            results.append({
                "question": answer.question,
                "user_answer": answer.user_answer,
                "correct_answer": answer.correct_answer,
                "partial_score": partial_score,
                "feedback": feedback,
                "question_type": answer.question_type,
            })
            print(f"Answer {i+1} partial score: {partial_score}, running score: {score}")

        feedback = f"You scored {score:.1f} out of {total}."
        if score >= total * 0.9:
            feedback += " Excellent work!"
        elif score >= total * 0.6:
            feedback += " Good effort, keep practicing!"
        else:
            feedback += " Review the material and try again."
        
        print(f"Final score: {score}/{total}")
        return {
            "score": score,
            "total": total,
            "results": results,
            "feedback": feedback
        }
    except Exception as e:
        print(f"Error in grade_answer: {str(e)}")
        return {"error": str(e)}

# Cleanup temporary directory on shutdown
@app.on_event("shutdown")
def cleanup():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)