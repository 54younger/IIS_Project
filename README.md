# Intelligent Behavioral Interview System

An intelligent behavioral interview system based on Google Gemini that dynamically adjusts question difficulty based on the interviewee's emotion, speech quality, and response completeness.

## Features

- 🎯 **Smart Question Selection**: Uses LLM to select the most contextually appropriate question from a curated question bank
- 🗣️ **Interactive Responses**: Provides immediate verbal acknowledgments during the interview for a natural conversational flow
- 📊 **Dynamic Difficulty Adjustment**: Adjusts question difficulty in real-time based on interviewee performance
- 😊 **Multi-dimensional Evaluation**: Considers emotion, speech quality, and response completeness
- 📈 **Visual Analysis**: Real-time display of difficulty trend
- 💬 **Personalized Feedback**: Provides constructive feedback for each answer
- 🔄 **No Duplicate Questions**: Tracks asked questions to ensure variety throughout the interview

## System Architecture

### Core Components

1. **BehavioralInterviewSystem**: Core interview logic
   - LLM-based contextual question selection
   - Interactive response generation
   - Difficulty adjustment algorithm
   - Answer evaluation
   - History management

2. **Evaluation Dimensions**
   - **Emotion**: Confident, Happy, Calm, Unconfident, Unhappy, Nervous
   - **Speech Quality**: Fluent, Clear, Hesitant, Unclear (evaluates fluency and clarity)
   - **Response Quality**: Complete, Good, Moderate, Brief, Unclear (evaluates completeness)

3. **Difficulty Levels**
   - **L1 (0-33)**: Entry-level behavioral questions
   - **L2 (33-67)**: Intermediate behavioral scenarios
   - **L3 (67-100)**: Advanced behavioral challenges

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Question Bank

Create a `questions.json` file in the project root with the following structure:

```json
{
  "L1": [
    {
      "id": "L1_Q1",
      "text": "Tell me about a time when you worked effectively in a team.",
      "robot_action": "nod",
      "emotion_trigger": "positive"
    }
  ],
  "L2": [...],
  "L3": [...]
}
```

### 3. Configure API Key

Get your Google Gemini API key:
- Visit https://makersuite.google.com/app/apikey
- Create an API key

Two ways to configure the key:

**Method 1: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

**Method 2: .env File**
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-api-key-here
```

**Method 3: Enter directly in the interface**
Enter the API key in the sidebar after launching the app

## Usage

### Web Interface Mode (Recommended)

1. Launch the app:
```bash
streamlit run interview_app.py
```

2. Open the displayed URL in your browser (usually http://localhost:8501)

3. Usage steps:
   - Enter Gemini API key in the sidebar (if not using .env)
   - Click "Initialize System"
   - Click "Start Interview"
   - Click "Generate New Question" to get an interview question
   - Enter your answer and select your performance (emotion, speech quality, response quality)
   - Submit to view:
     - Interactive response (immediate verbal acknowledgment)
     - Detailed feedback
     - Difficulty adjustment
   - Continue generating new questions for the interview

### Command Line Mode

```bash
python behavioral_interview.py
```

Follow the prompts to enter your API key and answer questions.

## Question Selection Algorithm

The system uses a two-step intelligent question selection process:

1. **Local Filtering**: Randomly select 2-3 candidate questions from the question bank based on current difficulty level
2. **LLM Selection**: Send candidates and interview history to Gemini to select the most contextually appropriate question

This ensures:
- Questions flow naturally from previous topics
- Different aspects of candidate experience are explored
- Good interview progression is maintained
- No duplicate questions are asked

## Difficulty Adjustment Algorithm

The system uses a weighted scoring algorithm to dynamically adjust difficulty:

```
Difficulty Adjustment = Emotion Score + Speech Score + Response Score

Emotion Score:
- Confident/Happy/Calm: +5
- Unconfident/Unhappy/Nervous: -5

Speech Quality Score:
- Fluent/Clear: +8
- Hesitant/Unclear: -8

Response Quality Score (Highest Weight):
- Complete: +15
- Good: +8
- Moderate: 0
- Brief: -10
- Unclear: -15
```

## Interface Features

### Main Interface

- **Interview Statistics**: Displays question count, current difficulty, difficulty level, etc.
- **Interview Q&A**: Shows questions and answer input area
- **Difficulty Trend Chart**: Visualizes difficulty changes
- **History**: View previous Q&A records

### Sidebar

- **System Configuration**: API key settings and system initialization
- **Interview Control**: Start, reset interview
- **Usage Instructions**: Quick operation guide

## Project Structure

```
IIS_Project/
├── behavioral_interview.py   # Core interview system
├── interview_app.py          # Streamlit web interface
├── questions.json            # Question bank (L1, L2, L3 levels)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API key)
└── README.md                 # This document
```

## Example Workflow

1. **Initial Stage** (Difficulty 20 - L1)
   - Question: "Tell me about a time when you worked effectively in a team."
   - Answer: Emotion-Confident, Speech-Fluent, Response-Complete
   - Interactive Response: "Great example!"
   - Result: Difficulty increased to 48 (+28) → L2

2. **Intermediate Stage** (Difficulty 48 - L2)
   - System selects from 3 L2 candidates based on context
   - Question: "Describe a situation where you had to manage conflicting priorities."
   - Answer: Emotion-Calm, Speech-Clear, Response-Good
   - Interactive Response: "I see, that makes sense."
   - Result: Difficulty increased to 69 (+21) → L3

3. **Advanced Stage** (Difficulty 69 - L3)
   - Question: "Tell me about a time you had to influence stakeholders without formal authority."
   - Answer: Emotion-Nervous, Speech-Hesitant, Response-Brief
   - Interactive Response: "Take your time."
   - Result: Difficulty decreased to 56 (-13) → L2

## Notes

1. **API Quota**: Gemini API may have usage limits, please be mindful of your quota
2. **Question Bank**: Ensure `questions.json` has sufficient questions for all three levels
3. **Honest Evaluation**: Please honestly evaluate your performance for the best interview experience
4. **Network Connection**: Requires a stable network connection to call the Gemini API
5. **LLM Usage**: The system uses LLM for question selection, interactive responses, and feedback generation

## FAQ

**Q: What if the API key is invalid?**
A: Please ensure you obtained a valid API key from Google AI Studio and check if it's correctly copied and pasted.

**Q: Question selection is slow?**
A: This depends on network conditions and Gemini API response time, usually takes a few seconds. The LLM analyzes 2-3 candidate questions to select the best one.

**Q: How to restart the interview?**
A: Click the "Reset Interview" button in the sidebar to clear history and start over.

**Q: What is the basis for difficulty adjustment?**
A: The system considers your emotion, speech quality, and response completeness, with response quality having the highest weight.

**Q: Will I get the same questions repeated?**
A: No, the system tracks all asked questions and ensures no duplicates during the same interview session.

## Tech Stack

- **AI Model**: Google Gemini 2.5 Flash Lite
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Environment Management**: python-dotenv
- **Language**: Python 3.8+

## Recent Updates

- ✅ Switched from random to LLM-based contextual question selection
- ✅ Added interactive verbal responses for natural conversation flow
- ✅ Renamed evaluation dimensions for clarity (Speech Quality, Response Quality)
- ✅ Implemented question bank system with no-repeat logic
- ✅ Upgraded to Gemini 2.5 Flash Lite model

## Future Improvements

- [ ] Integrate voice input functionality
- [ ] Integrate video emotion recognition for automatic emotion detection
- [ ] Add comprehensive interview report generation with analytics
- [ ] Support multiple interview types (technical interview, stress interview, etc.)
- [ ] Add practice mode and assessment mode
- [ ] Support multiple languages
- [ ] Expand question bank with more diverse scenarios

## License

This project is for learning and research purposes only.

## Contact

If you have any questions or suggestions, please feel free to open an Issue.
