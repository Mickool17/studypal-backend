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

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Study Companion Backend"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-questions")
async def generate_questions(request: QuestionRequest):
    try:
        print(f"Received request: {request}")
        num_questions = max(1, min(request.num_questions, 75))  # Updated to 75
        if request.question_type == "multiple_choice":
            prompt = (
                f"Generate exactly {num_questions} multiple-choice questions based on the following content:\n\n{request.text}\n\n"
                f"Provide 4 answer options per question, with the correct answer marked with **, e.g., c) **Correct Option**. "
                f"Ensure each question is numbered (e.g., Q1, Q2) and formatted clearly with a period after the question number (e.g., Q1.). "
                f"If the content is insufficient, generate as many relevant questions as possible up to {num_questions}."
            )
        else:  # theoretical
            prompt = (
                f"Generate exactly {num_questions} open-ended theoretical questions based on the following content:\n\n{request.text}\n\n"
                f"For each question, provide a sample answer prefixed with 'Sample Answer:'. "
                f"Format each question as: '**Q#: Question text**' and the sample answer as '**Sample Answer: Answer text**'. "
                f"Ensure each question is numbered (e.g., Q1, Q2). "
                f"If the content is insufficient, generate as many relevant questions as possible up to {num_questions}."
            )
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4000  # Adjusted for 75 questions
            }
        }
        response = requests.post(GEMINI_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        questions = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"Gemini response length: ${len(questions)} characters")
        
        # Count generated questions
        if request.question_type == "multiple_choice":
            question_matches = re.findall(r'Q\d+\.\s+', questions)
            question_count = len(question_matches)
        else:
            question_matches = re.findall(r'\*\*Q\d+:', questions)
            question_count = len(question_matches)
        print(f"Generated {question_count} questions out of {num_questions} requested")
        
        if question_count < num_questions:
            print(f"Warning: Generated fewer questions ({question_count}) than requested ({num_questions})")
        
        return {"questions": questions}
    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        return {"error": str(e)}

@app.post("/grade-answer")
async def grade_answer(request: GradeRequest):
    try:
        results = []
        score = 0.0
        total = len(request.answers)

        for answer in request.answers:
            if answer.question_type == "multiple_choice":
                is_correct = answer.user_answer.strip() == answer.correct_answer.strip()
                partial_score = 1.0 if is_correct else 0.0
                score += partial_score
                results.append({
                    "question": answer.question,
                    "user_answer": answer.user_answer,
                    "correct_answer": answer.correct_answer,
                    "partial_score": partial_score,
                    "feedback": "Correct answer!" if is_correct else "Incorrect, please review the material.",
                    "question_type": answer.question_type,
                })
            else:  # theoretical
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
                    print(f"JSON decode error: {str(e)} for response: {grading_text}")
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

        feedback = f"You scored {score:.1f} out of {total}."
        if score >= total * 0.9:
            feedback += " Excellent work!"
        elif score >= total * 0.6:
            feedback += " Good effort, keep practicing!"
        else:
            feedback += " Review the material and try again."

        return {
            "score": score,
            "total": total,
            "results": results,
            "feedback": feedback
        }
    except Exception as e:
        print(f"Error in grade_answer: {str(e)}")
        return {"error": str(e)}