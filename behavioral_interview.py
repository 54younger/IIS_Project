"""
Intelligent Behavioral Interview System Based on Gemini
Dynamically adjusts questions based on interviewee's emotion, answer quality, and correctness
"""

import google.generativeai as genai
import os
import json
import random
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Emotion(Enum):
    """Emotion types"""
    HAPPY = "Happy"
    CONFIDENT = "Confident"
    UNHAPPY = "Unhappy"
    UNCONFIDENT = "Unconfident"
    NERVOUS = "Nervous"
    CALM = "Calm"


class SpeechQuality(Enum):
    """Speech quality - fluency and clarity"""
    HESITANT = "Hesitant"
    FLUENT = "Fluent"
    UNCLEAR = "Unclear"
    CLEAR = "Clear"


class ResponseQuality(Enum):
    """Response quality and completeness"""
    COMPLETE = "Complete"
    GOOD = "Good"
    MODERATE = "Moderate"
    BRIEF = "Brief"
    UNCLEAR = "Unclear"


@dataclass
class InterviewResponse:
    """Interview response data structure"""
    answer: str
    emotion: Emotion
    speech_quality: SpeechQuality
    response_quality: ResponseQuality


@dataclass
class DifficultyLevel:
    """Difficulty level"""
    score: float  # 0-100, higher means harder
    
    def adjust(self, delta: float) -> 'DifficultyLevel':
        """Adjust difficulty"""
        new_score = max(0, min(100, self.score + delta))
        return DifficultyLevel(new_score)
    
    def get_level_name(self) -> str:
        """Get difficulty level name (L1, L2, L3)"""
        if self.score < 33:
            return "L1"
        elif self.score < 67:
            return "L2"
        else:
            return "L3"


class BehavioralInterviewSystem:
    """Intelligent Behavioral Interview System"""
    
    def __init__(self, api_key: str = None, questions_file: str = "questions.json"):
        """Initialize the system"""
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        else:
            raise ValueError("Please provide a Gemini API key")
        
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        self.difficulty = DifficultyLevel(20)  # Initial difficulty: L1 (start easy)
        self.interview_history: List[Dict] = []
        self.current_question = ""
        self.question_count = 0
        
        # Load questions from JSON file
        self.questions_file = questions_file
        self.questions = self._load_questions()
        self.asked_questions: Set[str] = set()  # Track asked questions to avoid repetition
    
    def _load_questions(self) -> Dict[str, List[Dict]]:
        """Load questions from JSON file"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
                # Ensure we have the expected structure
                for level in ['L1', 'L2', 'L3']:
                    if level not in questions_data:
                        questions_data[level] = []
                return questions_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Questions file '{self.questions_file}' not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{self.questions_file}'")
        
    def analyze_response(self, response: InterviewResponse) -> float:
        """
        Analyze interviewee's response and return difficulty adjustment value
        Positive value increases difficulty, negative value decreases difficulty
        """
        adjustment = 0.0
        
        # Adjust based on emotion
        if response.emotion in [Emotion.CONFIDENT, Emotion.HAPPY, Emotion.CALM]:
            adjustment += 4
        elif response.emotion in [Emotion.UNCONFIDENT, Emotion.UNHAPPY, Emotion.NERVOUS]:
            adjustment -= 4
            
        # Adjust based on speech quality (fluency and clarity)
        if response.speech_quality in [SpeechQuality.FLUENT, SpeechQuality.CLEAR]:
            adjustment += 6
        elif response.speech_quality in [SpeechQuality.HESITANT, SpeechQuality.UNCLEAR]:
            adjustment -= 6
            
        # Adjust based on response quality (highest weight)
        if response.response_quality == ResponseQuality.COMPLETE:
            adjustment += 10
        elif response.response_quality == ResponseQuality.GOOD:
            adjustment += 6
        elif response.response_quality == ResponseQuality.MODERATE:
            adjustment += 1
        elif response.response_quality == ResponseQuality.BRIEF:
            adjustment -= 6
        elif response.response_quality == ResponseQuality.UNCLEAR:
            adjustment -= 10
            
        return adjustment
    
    def generate_interactive_response(self, response: InterviewResponse) -> str:
        """Generate a brief, natural interactive response based on the interviewee's performance"""
        prompt = f"""You are a friendly, encouraging HR interviewer. Based on the candidate's performance, 
        give a brief, natural verbal response (1-2 words or a short phrase).

Interview Question: {self.current_question}
Candidate's Answer: {response.answer}

Candidate's Performance:
- Emotion: {response.emotion.value}
- Speech: {response.speech_quality.value}
- Completeness: {response.response_quality.value}

Guidelines:
- If confident/happy/good performance: Use positive acknowledgments like "Great!", "Interesting!", "I see", "Nice", "Cool"
- If nervous/hesitant/brief: Use encouraging phrases like "No worries", "That's okay", "Don't worry"
- If unclear: Use gentle prompts like "I understand", "Okay", "Got it"
- Keep it very short (1-5 words max)
- Sound natural and conversational
- Be warm and supportive

Respond with ONLY the brief phrase, nothing else:"""
        
        try:
            llm_response = self.model.generate_content(prompt)
            return llm_response.text.strip()
        except Exception as e:
            # Fallback responses based on emotion
            if response.emotion in [Emotion.NERVOUS, Emotion.UNCONFIDENT, Emotion.UNHAPPY]:
                return "Take your time."
            elif response.emotion in [Emotion.CONFIDENT, Emotion.HAPPY]:
                return "Great!"
            else:
                return "I see."
    
    def generate_question(self) -> str:
        """Select a question from the question bank using LLM for contextual relevance"""
        try:
            # Get difficulty level (L1, L2, or L3)
            level = self.difficulty.get_level_name()
            
            # Get available questions for this level (filter out already asked)
            available_questions = [q for q in self.questions.get(level, []) 
                                 if q.get('id') not in self.asked_questions]
            
            # If all questions at this level have been asked, try adjacent levels
            if not available_questions:
                if level == "L2":
                    # Try L1 and L3
                    for alt_level in ["L1", "L3"]:
                        available_questions = [q for q in self.questions.get(alt_level, [])
                                             if q.get('id') not in self.asked_questions]
                        if available_questions:
                            break
                elif level == "L1":
                    # Try L2
                    available_questions = [q for q in self.questions.get("L2", [])
                                         if q.get('id') not in self.asked_questions]
                else:  # L3
                    # Try L2
                    available_questions = [q for q in self.questions.get("L2", [])
                                         if q.get('id') not in self.asked_questions]
            
            # If still no questions available, reset the asked questions
            if not available_questions:
                self.asked_questions.clear()
                available_questions = self.questions.get(level, [])
            
            if not available_questions:
                return "No questions available in the question bank."
            
            # Step 1: Randomly select 2-3 candidate questions
            num_candidates = min(3, len(available_questions))
            candidates = random.sample(available_questions, num_candidates)
            
            # Step 2: Use LLM to select the most contextually appropriate question
            selected = self._select_question_with_llm(candidates)
            
            self.current_question = selected.get('text', selected.get('question', 'No question text'))
            self.asked_questions.add(selected.get('id'))
            self.question_count += 1
            return self.current_question
                
        except Exception as e:
            return f"Error selecting question: {str(e)}"
    
    def _select_question_with_llm(self, candidates: List[Dict]) -> Dict:
        """Use LLM to select the most contextually appropriate question from candidates"""
        if len(candidates) == 1:
            return candidates[0]
        
        # Build context from interview history
        context = ""
        if self.interview_history:
            recent_questions = [h['question'] for h in self.interview_history[-3:]]
            context = f"Previous questions asked:\n" + "\n".join([f"- {q}" for q in recent_questions])
        else:
            context = "This is the first question of the interview."
        
        # Build candidate list
        candidate_text = ""
        for i, candidate in enumerate(candidates, 1):
            candidate_text += f"{i}. {candidate.get('text', candidate.get('question', ''))}\n"
        
        prompt = f"""You are an HR interviewer conducting a behavioral interview. Based on the interview context, 
        select the most appropriate next question.

{context}

Candidate Questions:
{candidate_text}

Select the question that:
- Flows naturally from the previous questions (avoid repetitive themes)
- Explores different aspects of the candidate's experience
- Maintains good interview progression
- Feels contextually appropriate

Respond with ONLY the number (1, 2, or 3) of the best question:"""
        
        try:
            response = self.model.generate_content(prompt)
            choice = response.text.strip()
            
            # Parse the choice
            for char in choice:
                if char.isdigit():
                    index = int(char) - 1
                    if 0 <= index < len(candidates):
                        return candidates[index]
            
            # Fallback to first candidate if parsing fails
            return candidates[0]
            
        except Exception as e:
            # Fallback to random selection if LLM fails
            return random.choice(candidates)
    
    def evaluate_answer(self, user_answer: str, response: InterviewResponse) -> Dict:
        """Evaluate answer and generate feedback"""
        # Generate interactive response first (quick acknowledgment)
        interactive_response = self.generate_interactive_response(response)
        
        # Analyze response and adjust difficulty
        adjustment = self.analyze_response(response)
        old_difficulty = self.difficulty.score
        self.difficulty = self.difficulty.adjust(adjustment)
        
        # Record history
        self.interview_history.append({
            'question': self.current_question,
            'answer': response.answer,
            'emotion': response.emotion.value,
            'speech_quality': response.speech_quality.value,
            'response_quality': response.response_quality.value,
            'difficulty_before': old_difficulty,
            'difficulty_after': self.difficulty.score,
            'adjustment': adjustment,
            'interactive_response': interactive_response
        })
        
        # Generate evaluation feedback
        feedback = self.generate_feedback(response)
        
        return {
            'interactive_response': interactive_response,
            'feedback': feedback,
            'difficulty_change': adjustment,
            'new_difficulty': self.difficulty.score,
            'difficulty_level': self.difficulty.get_level_name()
        }
    
    def generate_feedback(self, response: InterviewResponse) -> str:
        """Generate personalized feedback using LLM"""
        prompt = f"""As a professional HR interviewer, please evaluate the following interview response:

Question: {self.current_question}
Answer: {response.answer}
Observed Performance: 
- Emotion/Confidence: {response.emotion.value}
- Speech Fluency: {response.speech_quality.value}
- Response Completeness: {response.response_quality.value}

Please provide brief, constructive feedback (2-3 sentences) that:
1. Acknowledges what the candidate did well
2. Offers one specific suggestion for improvement if needed
3. Is encouraging and professional

Keep the feedback under 60 words.
"""
        try:
            feedback_response = self.model.generate_content(prompt)
            return feedback_response.text.strip()
        except Exception as e:
            return "Thank you for your answer. Let's continue to the next question."
    
    def get_interview_summary(self) -> Dict:
        """Get interview summary"""
        if not self.interview_history:
            return {"message": "No interview records yet"}
        
        # Calculate statistics
        total_questions = len(self.interview_history)
        avg_difficulty = sum(r['difficulty_after'] for r in self.interview_history) / total_questions
        
        # Count emotion distribution
        emotions = [r['emotion'] for r in self.interview_history]
        speech_qualities = [r['speech_quality'] for r in self.interview_history]
        response_qualities = [r['response_quality'] for r in self.interview_history]
        
        return {
            'total_questions': total_questions,
            'average_difficulty': avg_difficulty,
            'final_difficulty': self.difficulty.score,
            'emotions': emotions,
            'speech_qualities': speech_qualities,
            'response_qualities': response_qualities,
            'difficulty_trend': [r['difficulty_after'] for r in self.interview_history]
        }
    
    def reset(self):
        """Reset interview"""
        self.difficulty = DifficultyLevel(20)  # Reset to L1
        self.interview_history = []
        self.current_question = ""
        self.question_count = 0
        self.asked_questions.clear()  # Clear asked questions for fresh start


def main():
    """Command line test interface"""
    print("=" * 60)
    print("Intelligent Behavioral Interview System")
    print("=" * 60)
    
    api_key = input("Please enter Gemini API key (or press Enter to use environment variable): ").strip()
    if not api_key:
        api_key = None
    
    try:
        system = BehavioralInterviewSystem(api_key)
        print("\nInterview starts!\n")
        
        while True:
            # Generate question
            question = system.generate_question()
            print(f"\n[Question {system.question_count}] (Difficulty: {system.difficulty.get_level_name()} - {system.difficulty.score:.1f})")
            print(question)
            
            # Get answer
            print("\nPlease enter your answer:")
            answer = input("> ").strip()
            
            if answer.lower() in ['quit', 'exit']:
                break
            
            # Get performance data
            print("\nPlease enter your emotion (Happy/Confident/Unhappy/Unconfident/Nervous/Calm):")
            emotion_input = input("> ").strip()
            
            print("Please enter speech quality (Hesitant/Fluent/Unclear/Clear):")
            speech_quality_input = input("> ").strip()
            
            print("Please enter response quality (Complete/Good/Moderate/Brief/Unclear):")
            response_quality_input = input("> ").strip()
            
            # Map input to enum
            emotion_map = {
                "Happy": Emotion.HAPPY,
                "Confident": Emotion.CONFIDENT,
                "Unhappy": Emotion.UNHAPPY,
                "Unconfident": Emotion.UNCONFIDENT,
                "Nervous": Emotion.NERVOUS,
                "Calm": Emotion.CALM
            }
            
            speech_quality_map = {
                "Hesitant": SpeechQuality.HESITANT,
                "Fluent": SpeechQuality.FLUENT,
                "Unclear": SpeechQuality.UNCLEAR,
                "Clear": SpeechQuality.CLEAR
            }
            
            response_quality_map = {
                "Complete": ResponseQuality.COMPLETE,
                "Good": ResponseQuality.GOOD,
                "Moderate": ResponseQuality.MODERATE,
                "Brief": ResponseQuality.BRIEF,
                "Unclear": ResponseQuality.UNCLEAR
            }
            
            response = InterviewResponse(
                answer=answer,
                emotion=emotion_map.get(emotion_input, Emotion.CALM),
                speech_quality=speech_quality_map.get(speech_quality_input, SpeechQuality.CLEAR),
                response_quality=response_quality_map.get(response_quality_input, ResponseQuality.MODERATE)
            )
            
            # Evaluate answer
            evaluation = system.evaluate_answer(answer, response)
            
            print(f"\n[Interactive Response] {evaluation['interactive_response']}")
            print(f"\n[Feedback] {evaluation['feedback']}")
            print(f"Difficulty adjustment: {evaluation['difficulty_change']:+.1f} -> New difficulty: {evaluation['difficulty_level']} ({evaluation['new_difficulty']:.1f})")
            
            print("\n" + "-" * 60)
            
            # Ask whether to continue
            continue_input = input("\nContinue to next question? (y/n): ").strip().lower()
            if continue_input != 'y':
                break
        
        # Display summary
        print("\n" + "=" * 60)
        print("Interview Summary")
        print("=" * 60)
        summary = system.get_interview_summary()
        print(f"Total questions: {summary['total_questions']}")
        print(f"Average difficulty: {summary['average_difficulty']:.1f}")
        print(f"Final difficulty: {summary['final_difficulty']:.1f}")
        print("\nThank you for participating in the interview!")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
