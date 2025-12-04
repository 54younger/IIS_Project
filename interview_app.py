"""
Intelligent Behavioral Interview System - Streamlit Web Interface
Provides a visual interview interaction experience
"""

import streamlit as st
from behavioral_interview import (
    BehavioralInterviewSystem,
    InterviewResponse,
    Emotion,
    SpeechQuality,
    ResponseQuality
)
import plotly.graph_objects as go
import os


def init_session_state():
    """Initialize session state"""
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False
    if 'last_feedback' not in st.session_state:
        st.session_state.last_feedback = ""


def plot_difficulty_trend():
    """Plot difficulty trend chart"""
    if st.session_state.system and st.session_state.system.interview_history:
        history = st.session_state.system.interview_history
        
        questions = [f"Q{i+1}" for i in range(len(history))]
        difficulties = [record['difficulty_after'] for record in history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=questions,
            y=difficulties,
            mode='lines+markers',
            name='Difficulty Trend',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        
        # Add difficulty level reference lines
        fig.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Basic")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Elementary")
        fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Intermediate")
        
        fig.update_layout(
            title="Interview Difficulty Trend",
            xaxis_title="Question Number",
            yaxis_title="Difficulty Score",
            yaxis_range=[0, 100],
            height=400
        )
        
        return fig
    return None


def display_statistics():
    """Display statistics"""
    if st.session_state.system and st.session_state.system.interview_history:
        summary = st.session_state.system.get_interview_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", summary['total_questions'])
        
        with col2:
            st.metric("Current Difficulty", f"{summary['final_difficulty']:.1f}")
        
        with col3:
            difficulty_level = st.session_state.system.difficulty.get_level_name()
            st.metric("Difficulty Level", difficulty_level)
        
        with col4:
            st.metric("Average Difficulty", f"{summary['average_difficulty']:.1f}")


def main():
    st.set_page_config(
        page_title="Intelligent Behavioral Interview System",
        page_icon="💼",
        layout="wide"
    )
    
    init_session_state()
    
    # Title
    st.title("💼 Intelligent Behavioral Interview System")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # API key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get('GEMINI_API_KEY', ''),
            help="Enter your Gemini API key or set GEMINI_API_KEY environment variable"
        )
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            try:
                st.session_state.system = BehavioralInterviewSystem(api_key if api_key else None)
                st.success("✅ System initialized successfully!")
                st.session_state.interview_started = False
                st.session_state.waiting_for_answer = False
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
        
        st.markdown("---")
        
        # Interview control
        st.header("📋 Interview Control")
        
        if st.session_state.system:
            if not st.session_state.interview_started:
                if st.button("▶️ Start Interview", use_container_width=True):
                    st.session_state.interview_started = True
                    st.session_state.waiting_for_answer = False
                    st.rerun()
            else:
                if st.button("🔄 Reset Interview", use_container_width=True):
                    st.session_state.system.reset()
                    st.session_state.interview_started = False
                    st.session_state.current_question = ""
                    st.session_state.waiting_for_answer = False
                    st.session_state.last_feedback = ""
                    st.success("✅ Interview has been reset")
                    st.rerun()
        
        st.markdown("---")
        
        # Usage instructions
        st.header("📖 Usage Instructions")
        st.markdown("""
        1. Enter Gemini API key
        2. Click "Initialize System"
        3. Click "Start Interview"
        4. Answer questions and fill in performance
        5. System will automatically adjust difficulty
        """)
    
    # Main content area
    if not st.session_state.system:
        st.info("👈 Please configure and initialize the system in the sidebar first")
        return
    
    if not st.session_state.interview_started:
        st.info("👈 Please click the 'Start Interview' button in the sidebar to begin")
        
        # Display system description
        st.markdown("""
        ### Welcome to the Intelligent Behavioral Interview System
        
        This system dynamically adjusts question difficulty based on your performance:
        - 📊 **Emotion Analysis**: Confident, happy expressions will increase difficulty
        - 💬 **Answer Quality**: Fluent, clear answers will raise difficulty
        - ✅ **Correctness Evaluation**: Correct answers will increase difficulty, incorrect answers will decrease it
        
        Each question should be answerable within **30 seconds**.
        """)
        return
    
    # Display statistics
    st.subheader("📊 Interview Statistics")
    display_statistics()
    st.markdown("---")
    
    # Main interview area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("💬 Interview Q&A")
        
        # Generate or display question
        if not st.session_state.waiting_for_answer:
            if st.button("🎯 Generate New Question", type="primary"):
                with st.spinner("Generating question..."):
                    question = st.session_state.system.generate_question()
                    st.session_state.current_question = question
                    st.session_state.waiting_for_answer = True
                    st.session_state.last_feedback = ""
                    st.rerun()
        
        # Display current question
        if st.session_state.current_question:
            difficulty_info = f"{st.session_state.system.difficulty.get_level_name()} (Difficulty: {st.session_state.system.difficulty.score:.1f})"
            st.info(f"**Question {st.session_state.system.question_count}** - {difficulty_info}")
            st.markdown(f"### {st.session_state.current_question}")
            
            # Display previous feedback
            if st.session_state.last_feedback:
                st.success(f"**Feedback:** {st.session_state.last_feedback}")
        
        # Answer input area
        if st.session_state.waiting_for_answer:
            st.markdown("---")
            st.markdown("#### Please enter your answer")
            
            with st.form("answer_form"):
                # Answer content
                answer = st.text_area(
                    "Your Answer",
                    height=150,
                    placeholder="Please enter your answer (should be answerable within 30 seconds)..."
                )
                
                # Performance evaluation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    emotion = st.selectbox(
                        "😊 Emotion",
                        ["Confident", "Happy", "Calm", "Unconfident", "Unhappy", "Nervous"]
                    )
                
                with col2:
                    speech_quality = st.selectbox(
                        "💬 Speech Fluency",
                        ["Fluent", "Clear", "Hesitant", "Unclear"]
                    )
                
                with col3:
                    response_quality = st.selectbox(
                        "✅ Response Completeness",
                        ["Complete", "Good", "Moderate", "Brief", "Unclear"]
                    )
                
                submitted = st.form_submit_button("📤 Submit Answer", type="primary", use_container_width=True)
                
                if submitted:
                    if not answer.strip():
                        st.error("Please enter your answer!")
                    else:
                        # Map input
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
                            emotion=emotion_map[emotion],
                            speech_quality=speech_quality_map[speech_quality],
                            response_quality=response_quality_map[response_quality]
                        )
                        
                        with st.spinner("Evaluating your answer..."):
                            evaluation = st.session_state.system.evaluate_answer(answer, response)
                            st.session_state.last_feedback = evaluation['feedback']
                            st.session_state.waiting_for_answer = False
                            
                            # Display interactive response first (like a quick reaction)
                            st.info(f"💬 **{evaluation['interactive_response']}**")
                            
                            # Display evaluation results
                            if evaluation['difficulty_change'] > 0:
                                st.success(f"✨ Great! Difficulty increased by {evaluation['difficulty_change']:.1f} points")
                            elif evaluation['difficulty_change'] < 0:
                                st.info(f"💪 Difficulty decreased by {abs(evaluation['difficulty_change']):.1f} points, keep going!")
                            else:
                                st.info("➡️ Difficulty remains unchanged")
                            
                            st.rerun()
    
    with col_right:
        st.subheader("📈 Difficulty Trend")
        
        # Plot difficulty trend
        fig = plot_difficulty_trend()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Difficulty trend will be displayed after answering questions")
        
        # Display history
        if st.session_state.system.interview_history:
            st.markdown("---")
            st.subheader("📝 History")
            
            for i, record in enumerate(reversed(st.session_state.system.interview_history[-5:]), 1):
                idx = len(st.session_state.system.interview_history) - i + 1
                with st.expander(f"Question {idx} ({record['difficulty_after']:.1f})"):
                    st.markdown(f"**Q:** {record['question']}")
                    st.markdown(f"**A:** {record['answer'][:100]}...")
                    st.caption(f"Emotion: {record['emotion']} | Speech: {record['speech_quality']} | Completeness: {record['response_quality']}")
                    
                    if record['adjustment'] > 0:
                        st.success(f"Difficulty +{record['adjustment']:.1f}")
                    elif record['adjustment'] < 0:
                        st.warning(f"Difficulty {record['adjustment']:.1f}")


if __name__ == "__main__":
    main()
