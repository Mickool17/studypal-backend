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
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"

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
    # Replace multiple spaces, tabs, or newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Fix concatenated words (e.g., human–computersystems -> human-computer systems)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Normalize dashes and special characters
    text = text.replace('–', '-').replace('’', "'")
    # Ensure proper spacing around punctuation
    text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)
    # Remove stray quotation marks
    text = re.sub(r'"+', '', text)
    # Trim leading/trailing whitespace
    text = text.strip()
    return text

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
        
        # Preprocess extracted text
        text = preprocess_text(text)
        print(f"Extracted text length: {len(text)}")
        return {"text": text}
    except Exception as e:
        print(f"Error in extract_text: {str(e)}")
        return {"error": str(e)}

@app.post("/generate-questions")
async def generate_questions(request: QuestionRequest):
    try:
        print(f"Received request: {request}")
        num_questions = max(1, min(request.num_questions, 75))
        max_attempts = 3
        parsed_questions = []

        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1} to generate questions")
            if request.question_type == "multiple_choice":
                prompt = (
                    f"Generate exactly {num_questions} multiple-choice questions based on the following content:\n\n{request.text}\n\n"
                    f"Provide 4 answer options per question, with the correct answer marked with **, e.g., c) **Correct Option**. "
                    f"Ensure each option has non-empty text after the prefix (e.g., a), b)). "
                    f"Ensure each question is numbered (e.g., Q1, Q2) and formatted clearly with a period after the question number (e.g., Q1.). "
                    f"If the content is insufficient, generate relevant questions based on the topic to reach exactly {num_questions} questions."
                )
            else:  # theoretical
                prompt = (
                    f"Generate exactly {num_questions} open-ended theoretical questions based on the following content:\n\n{request.text}\n\n"
                    f"For each question, provide a sample answer prefixed with 'Sample Answer:'. "
                    f"Format each question as: '**Q#: Question text**' and the sample answer as: '**Sample Answer: Answer text**'. "
                    f"Ensure each question is numbered (e.g., Q1, Q2). "
                    f"Avoid including stray quotation marks (e.g., \"\") in the question text or sample answer. "
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
            print(f"Attempt {attempt + 1} - Gemini raw response length: {len(questions)} characters")
            print(f"Attempt {attempt + 1} - Gemini raw response: {questions[:500]}...")

            # Parse questions
            parsed_questions = []
            if request.question_type == "multiple_choice":
                question_matches = re.findall(r'(Q\d+\.\s+.*?)(?=Q\d+\.|$)', questions, re.DOTALL)
                question_count = len(question_matches)
                print(f"Attempt {attempt + 1} - Parsed {question_count} multiple-choice questions")
                for q in question_matches:
                    lines = q.strip().split('\n')
                    question_text = lines[0].replace('Q', '', 1).lstrip('0123456789. ').replace('""', '').strip()
                    # Extract options, removing markers and extra text
                    options = []
                    raw_options = lines[1:5] if len(lines) >= 5 else lines[1:]
                    for line in raw_options:
                        line = line.strip()
                        if line:
                            # Remove ** and "Correct Option" or similar text
                            clean_option = re.sub(r'\s*\*\*.*?(Correct Option)?\s*$', '', line).strip()
                            # Ensure option starts with a letter and has non-empty text
                            if re.match(r'^[a-d]\)\s*\S+', clean_option):
                                options.append(clean_option)
                    # Validate exactly 4 non-empty options
                    if len(options) != 4:
                        print(f"Attempt {attempt + 1} - Warning: Question '{question_text}' has {len(options)} options, expected 4: {options}")
                        continue
                    # Find correct answer by matching the option with ** in raw text
                    correct_option = None
                    for line in raw_options:
                        if '**' in line:
                            clean_correct = re.sub(r'\s*\*\*.*?(Correct Option)?\s*$', '', line).strip()
                            if clean_correct in options:
                                correct_option = clean_correct
                                break
                    if not correct_option or not re.match(r'^[a-d]\)\s*\S+', correct_option):
                        print(f"Attempt {attempt + 1} - Warning: Invalid or missing correct option for question '{question_text}', options: {options}")
                        continue
                    parsed_questions.append({
                        'text': question_text,
                        'type': 'multiple_choice',
                        'options': options,
                        'correct_answer': correct_option
                    })
            else:  # theoretical
                question_matches = re.findall(r'\*\*Q\d+:.*?\*\*\n.*?\n\*\*Sample Answer:.*?(?=\*\*Q\d+:|$)', questions, re.DOTALL)
                question_count = len(question_matches)
                print(f"Attempt {attempt + 1} - Parsed {question_count} theoretical questions")
                for q in question_matches:
                    lines = q.strip().split('\n')
                    question_text = lines[0].replace('**Q', '').lstrip('0123456789: ').replace('**', '').replace('""', '').strip()
                    # Additional sanitization for stray quotation marks
                    question_text = re.sub(r'"+', '', question_text).strip()
                    sample_answer = lines[2].replace('**Sample Answer:', '').replace('""', '').strip()
                    sample_answer = re.sub(r'"+', '', sample_answer).strip()
                    # Validate non-empty question text
                    if not question_text:
                        print(f"Attempt {attempt + 1} - Warning: Empty question text for theoretical question, skipping")
                        continue
                    parsed_questions.append({
                        'text': question_text,
                        'type': 'theoretical',
                        'correct_answer': sample_answer
                    })

            print(f"Attempt {attempt + 1} - Generated {len(parsed_questions)} questions out of {num_questions} requested")
            if len(parsed_questions) >= num_questions:
                break  # Exit if we have enough questions
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1} - Retrying due to insufficient valid questions ({len(parsed_questions)} < {num_questions})")

        if len(parsed_questions) < num_questions:
            print(f"Warning: Generated fewer valid questions ({len(parsed_questions)}) than requested ({num_questions}) after {max_attempts} attempts")

        return {"questions": parsed_questions[:num_questions]}
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
            print(f"User answer: '{answer.user_answer}'")
            print(f"Correct answer: '{answer.correct_answer}'")
            partial_score = 0.0
            feedback = ""
            if answer.question_type == "multiple_choice":
                user_answer = re.sub(r'^[a-d]\)\s*', '', answer.user_answer.strip()).lower()
                correct_answer = re.sub(r'^[a-d]\)\s*', '', answer.correct_answer.strip()).lower()
                is_correct = user_answer == correct_answer
                partial_score = 1.0 if is_correct else 0.0
                feedback = "Correct answer!" if is_correct else f"Incorrect, the correct answer was: {answer.correct_answer}"
                print(f"Normalized user answer: '{user_answer}'")
                print(f"Normalized correct answer: '{correct_answer}'")
                print(f"Is correct: {is_correct}")
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