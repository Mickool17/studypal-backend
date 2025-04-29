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

def preprocess_text(text: str) -> str:
    """Clean and normalize text to improve API response quality."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('–', '-').replace('’', "'")
    text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)
    return text.strip()

def extract_mcq_question(block: str) -> dict:
    """Extract a complete multiple choice question from a text block."""
    lines = [line.strip() for line in block.split('\n') if line.strip()]
    if not lines:
        return None
    
    # Extract question text
    question_text = re.sub(r'^Q\d+\.\s*', '', lines[0]).strip()
    
    # Process options
    options = []
    correct_answer = None
    
    for line in lines[1:]:
        # Match option pattern (a), b), etc.) with correct answer markers
        match = re.match(r'^([a-d])\)\s*(\*\*)?(.*?)(\*\*)?$', line, re.IGNORECASE)
        if match:
            letter = match.group(1).lower()
            is_correct = match.group(2) is not None or match.group(4) is not None
            option_text = match.group(3).strip()
            
            # Handle empty options
            if not option_text:
                option_text = f"Correct Answer" if is_correct else f"Option {letter}"
            
            full_option = f"{letter}) {option_text}"
            options.append(full_option)
            
            if is_correct:
                correct_answer = full_option
    
    # Ensure we have exactly 4 options
    while len(options) < 4:
        letter = chr(97 + len(options))  # a, b, c, d
        options.append(f"{letter}) [Additional Option]")
    
    # If no correct answer marked, pick the first one and mark it
    if not correct_answer and options:
        correct_answer = options[0]
        options[0] = options[0].replace(')', ") [Likely Correct]")
        correct_answer = options[0]
    
    return {
        'text': question_text,
        'type': 'multiple_choice',
        'options': options,
        'correct_answer': correct_answer
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Study Companion Backend"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.lower().split('.')[-1]
        text = ""
        
        if file_extension == 'pdf':
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + " "
        elif file_extension == 'txt':
            text = await file.read()
            text = text.decode('utf-8')
        elif file_extension == 'docx':
            return {"error": "DOCX extraction must be handled client-side"}
        else:
            return {"error": f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt"}
        
        text = preprocess_text(text)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-questions")
async def generate_questions(request: QuestionRequest):
    try:
        num_questions = max(1, min(request.num_questions, 75))
        max_attempts = 3
        parsed_questions = []

        for attempt in range(max_attempts):
            if request.question_type == "multiple_choice":
                prompt = (
                    f"Generate exactly {num_questions} multiple-choice questions based on this content:\n\n{request.text}\n\n"
                    f"FORMAT REQUIREMENTS:\n"
                    f"1. Each question must start with 'Q1.', 'Q2.', etc. followed by the question text\n"
                    f"2. Provide exactly 4 options per question labeled a), b), c), d)\n"
                    f"3. Mark ONLY the correct answer with double asterisks (**) like: c) **Correct Answer**\n"
                    f"4. Never leave options empty - always provide option text\n"
                    f"5. Example format:\n"
                    f"Q1. What is the capital of France?\n"
                    f"a) London\n"
                    f"b) Berlin\n"
                    f"c) **Paris**\n"
                    f"d) Madrid\n\n"
                    f"If content is insufficient, generate relevant questions to reach exactly {num_questions} questions."
                )
            else:  # theoretical
                prompt = (
                    f"Generate exactly {num_questions} open-ended questions based on:\n\n{request.text}\n\n"
                    f"Format each as:\n"
                    f"**Q1: Question text**\n"
                    f"**Sample Answer: Answer text**\n\n"
                    f"Ensure questions are numbered sequentially (Q1, Q2, etc.)."
                )

            headers = {"Content-Type": "application/json"}
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

            # Parse questions based on type
            if request.question_type == "multiple_choice":
                question_blocks = re.findall(r'(Q\d+\.\s+.*?(?:\n[a-d]\)\s+.*?)+)(?=Q\d+\.|$)', questions, re.DOTALL)
                
                for block in question_blocks:
                    question_data = extract_mcq_question(block)
                    if question_data:
                        parsed_questions.append(question_data)
            else:  # theoretical
                question_matches = re.findall(r'\*\*Q\d+:.*?\*\*\n.*?\n\*\*Sample Answer:.*?(?=\*\*Q\d+:|$)', questions, re.DOTALL)
                for q in question_matches:
                    lines = q.strip().split('\n')
                    question_text = lines[0].replace('**Q', '').split(':', 1)[1].replace('**', '').strip()
                    sample_answer = lines[2].replace('**Sample Answer:', '').replace('**', '').strip()
                    parsed_questions.append({
                        'text': question_text,
                        'type': 'theoretical',
                        'correct_answer': sample_answer
                    })

            if len(parsed_questions) >= num_questions:
                break  # Exit if we have enough questions

        return {"questions": parsed_questions[:num_questions]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/grade-answer")
async def grade_answer(request: GradeRequest):
    try:
        results = []
        score = 0.0
        
        for answer in request.answers:
            partial_score = 0.0
            feedback = ""
            
            if answer.question_type == "multiple_choice":
                # Extract just the option letter from user's answer
                user_letter = re.sub(r'^([a-d])\)\s*.*$', r'\1', answer.user_answer.strip().lower())
                correct_letter = re.sub(r'^([a-d])\)\s*.*$', r'\1', answer.correct_answer.strip().lower())
                is_correct = user_letter == correct_letter
                partial_score = 1.0 if is_correct else 0.0
                feedback = "Correct!" if is_correct else f"Incorrect. The correct answer was: {answer.correct_answer}"
            else:
                if not answer.user_answer.strip():
                    feedback = "No answer provided."
                else:
                    prompt = (
                        f"Evaluate this answer for the question:\n{answer.question}\n\n"
                        f"Expected Answer: {answer.correct_answer}\n"
                        f"User Answer: {answer.user_answer}\n\n"
                        f"Provide a percentage score (0-100) and brief feedback. "
                        f"Return JSON format: {{\"score\": number, \"feedback\": \"string\"}}"
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
                    
                    try:
                        if '```json' in grading_text:
                            grading_text = grading_text.split('```json')[1].split('```')[0]
                        grading_data = json.loads(grading_text)
                        partial_score = min(max(float(grading_data.get("score", 0)) / 100, 1.0)
                        feedback = grading_data.get("feedback", "Evaluation unavailable")
                    except (json.JSONDecodeError, ValueError):
                        partial_score = 0.0
                        feedback = "Unable to evaluate this answer"
            
            score += partial_score
            results.append({
                "question": answer.question,
                "user_answer": answer.user_answer,
                "correct_answer": answer.correct_answer,
                "partial_score": partial_score,
                "feedback": feedback,
                "question_type": answer.question_type,
            })

        total = len(request.answers)
        overall_feedback = f"Score: {score:.1f}/{total} - "
        if score >= total * 0.9:
            overall_feedback += "Excellent!"
        elif score >= total * 0.7:
            overall_feedback += "Good job!"
        else:
            overall_feedback += "Keep practicing!"
        
        return {
            "score": score,
            "total": total,
            "results": results,
            "feedback": overall_feedback
        }
    except Exception as e:
        return {"error": str(e)}