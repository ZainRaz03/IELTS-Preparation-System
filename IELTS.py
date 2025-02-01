import os
from phi.agent import Agent
from phi.model.google import Gemini
import chromadb
from chromadb.config import Settings
import PyPDF2
import google.generativeai as genai
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np
from datetime import datetime
import random
import sounddevice as sd
import wavio
import base64
from dotenv import load_dotenv
load_dotenv()

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    

os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv('API_KEY'))

# Optional if using VOSK
# AudioSegment.converter = "C:/Users/IT USER/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
# AudioSegment.ffmpeg = "C:/Users/IT USER/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
# AudioSegment.ffprobe = "C:/Users/IT USER/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"

gemini_model = Gemini(id="gemini-1.5-flash")

client = chromadb.Client(Settings(persist_directory="./chromadb"))
try:
    collection = client.get_collection("ielts_passages")
except:
    collection = client.create_collection(name="ielts_passages")

class BaseAgent(Agent):
    def __init__(self, name, model):
        super().__init__(name=name, model=model)
        self.description="Base agent for IELTS evaluation system providing common functionality.",
        self.instructions=[
            "Process requests with consistent formatting",
            "Handle errors gracefully",
            "Provide clear and structured responses"
        ]

    def print_response(self, message):
        return super().print_response(message)

class IELTSReadingAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.passages = []
        self.description="Specialized agent for processing IELTS reading passages and managing content.",
        self.instructions=[
            "Extract text from PDF files maintaining formatting",
            "Clean and process text to remove unwanted characters and formatting",
            "Split content into chunks of approximately 500 words",
            "Maintain a collection of passages for testing",
            "Provide random passage selection when requested"
        ]

    def process_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                text = ' '.join(text.split())
                words = text.split()
                chunks = []
                current_chunk = []
                word_count = 0
                
                for word in words:
                    current_chunk.append(word)
                    word_count += 1
                    
                    if word_count >= 500:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        word_count = 0
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                self.passages.extend(chunks)
        
        return self.get_random_passage()

    def get_random_passage(self):
        if self.passages:
            return random.choice(self.passages)
        return None

class QuestionGeneratorAgentReading(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS question creator focused on generating reading comprehension questions.",
        self.instructions=[
            "Generate exactly 3 IELTS-style questions per passage",
            "Ensure questions test different aspects of comprehension",
            "Create questions that match official IELTS difficulty levels",
            "Format questions consistently with IELTS standards",
            "Avoid questions that can be answered without understanding the passage"
        ]

    def generate_questions(self, passage):
        prompt = f"""
        Generate exactly 3 IELTS-style reading comprehension questions based on this passage:
        {passage}
        
        Format each question like this:
        Question 1: [First question text]
        Question 2: [Second question text]
        Question 3: [Third question text]
        """
        response = self.run(prompt)
        questions = []
        for line in response.content.split('\n'):
            if line.startswith('Question'):
                questions.append(line.split(': ')[1])
        return questions
class AnswerEvaluationAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS evaluator providing detailed assessment of reading answers.",
        self.instructions=[
            "Compare user answers against passage content",
            "Provide clear correct/incorrect status for each answer",
            "Give detailed explanation for why an answer is correct or incorrect",
            "Suggest improvements for incorrect answers",
            "Format evaluation results for clear presentation in Streamlit"
        ]

    def evaluate_answers(self, passage, questions, user_answers):
        prompt = f"""
        Based on this passage:
        {passage}

        Evaluate these answers:
        Question 1: {questions[0]}
        User Answer 1: {user_answers[0]}

        Question 2: {questions[1]}
        User Answer 2: {user_answers[1]}

        Question 3: {questions[2]}
        User Answer 3: {user_answers[2]}

        For each answer, provide separately:
        1. Result whther correct or not
        2. The correct answer
        3. Brief explanation

        Provide detailed scoring and feedback on:

        1. Reading Comprehension (Score out of 9)
        Criteria: Understanding main ideas, grasping specific details, making inferences, recognizing author's purpose/tone.
        Evidence: Provide specific examples from responses that illustrate comprehension level.

        2. Answer Accuracy (Score out of 9)
        Criteria: Correctness of information, completeness of response, relevance to question, precision of details.
        Evidence: Support score with examples from each answer.

        3. Response Quality (Score out of 9)
        Criteria: Clarity of expression, use of evidence from text, logical reasoning, appropriate detail level.
        Evidence: Include specific examples demonstrating quality.

        4. Overall Performance (Score out of 9)
        Criteria: Consistency across answers, pattern recognition, critical thinking skills, evidence of time management.
        Evidence: Explain scoring with concrete examples.

        5. Final Band Score (Calculate average of above)
        Description: Explain overall performance level.

        Each Criterion:
        Justify: Score with specific examples from answers.
        Strengths: Highlight what was done well.
        Improvement: Provide detailed suggestions.

        """
        response = self.run(prompt)
        # formatted_response = self.format_evaluation_for_streamlit(response.content)
        return response.content


class AudioProcessingAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Specialized audio processing agent for IELTS listening tests.",
        self.instructions=[
            "Handle various audio file formats",
            "Convert speech to accurate text transcription",
            "Process audio with appropriate sampling rates",
            "Handle audio processing errors gracefully",
            "Ensure high-quality transcription for evaluation"
        ]
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_path):
        with sr.AudioFile("C:/Users/IT USER/Desktop/Multi Agent System/Ielts.wav") as source:
            audio_data = self.recognizer.record(source)
            try:
                transcript = self.recognizer.recognize_google(audio_data)
                return transcript
            except Exception as e:
                return f"Error transcribing audio: {str(e)}"

class ListeningEvaluationAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS listening test evaluator providing detailed assessment.",
        self.instructions=[
            "Compare user answers against audio transcript",
            "Evaluate accuracy of responses",
            "Provide detailed explanation for each answer",
            "Consider acceptable variations in answers",
            "Format evaluation results clearly",
            "Provide improvement suggestions for incorrect answers"
        ]


    def evaluate_listening_answers(self, transcript, user_answers):
        prompt = f"""
        Based on this audio transcript:
        {transcript}

        Evaluate these answers:
        Answer 1: {user_answers[0]}
        Answer 2: {user_answers[1]}
        Answer 3: {user_answers[2]}

        For each answer, provide separately:
        1. Result whther correct or not
        2. The correct answer
        3. Brief explanation

        Provide detailed scoring and feedback on:

        1. Listening Comprehension (Score out of 9)
        Criteria: Understanding of main points, recognition of specific details, grasp of speaker intentions, understanding of context.
        Evidence: Support with specific examples from responses.

        2. Information Accuracy (Score out of 9)
        Criteria: Correctness of facts, completeness of details, precision in numbers/dates/names, relevance of information.
        Evidence: Provide examples from each answer.

        3. Note-Taking Effectiveness (Score out of 9)
        Criteria: Key point identification, detail retention, priority understanding, information organization.
        Evidence: Include specific examples from responses.

        4. Response Quality (Score out of 9)
        Criteria: Answer completeness, detail accuracy, information relevance, expression clarity.
        Evidence: Support with concrete examples.

        5. Final Band Score (Calculate average of above)
        Description: Explain overall performance level.

        Each Criterion:
        Justify: Score with specific examples.
        Strengths: Highlight what was done well.
        Improvement: Provide detailed strategies.
        Practice: Suggest specific techniques.


        Format each response with clear sections and bullet points.
        """
        response = self.run(prompt)
        # formatted_response = self.format_listening_evaluation(response.content)
        return response.content

class WritingPromptAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS writing prompt generator focusing on Task 2 essays.",
        self.instructions=[
            "Generate contemporary and relevant essay topics",
            "Include clear task instructions",
            "Ensure topics are controversial enough for discussion",
            "Follow official IELTS Task 2 format",
            "Include word count requirement (250 words)",
            "Create topics suitable for academic discussion"
        ]
    
    def generate_writing_prompt(self):
        prompt = """
        Generate a new IELTS Writing Task prompt. Follow this format:
        
        1. Topic statement that presents two contrasting views on a contemporary issue
        2. Task instruction to discuss both views and give opinion
        3. Additional instruction about examples and reasoning
        4. Word count requirement (250 words)
        
        The topic should be suitable for IELTS, controversial enough for discussion, 
        and related to themes like: education, technology, society, environment, 
        health, work, or government.
        
        Format the response exactly like this:
        Write about the following topic:
        [Topic statement with contrasting views]
        
        Discuss both views and give your own opinion.
        
        Give reasons for your answer and include any relevant examples from your own knowledge or experience.
        
        Write at least 250 words.
        """
        response = self.run(prompt)
        return response.content

class WritingEvaluationAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS writing evaluator providing comprehensive assessment.",
        self.instructions=[
            "Evaluate fluency and coherence (0-9 band)",
            "Assess lexical resource and vocabulary usage",
            "Analyze grammatical range and accuracy",
            "Evaluate pronunciation quality",
            "Provide detailed feedback for each criterion",
            "Calculate overall band score",
            "Suggest specific improvements for each area"
        ]

    def evaluate_writing(self, text, prompt):
        evaluation_prompt = f"""
        Evaluate this IELTS Writing Task response for the following prompt:
        
        {prompt}
        
        Student's response:
        {text}
        
        Provide detailed scoring and feedback on:
        1. Task Achievement (out of 9)
        2. Coherence and Cohesion (out of 9)
        3. Lexical Resource (out of 9)
        4. Grammatical Range and Accuracy (out of 9)
        5. Overall Band Score (average of above)
        
        For each criterion, explain the score with specific examples from the text.
        Also provide suggestions for improvement.
        """
        response = self.run(evaluation_prompt)
        return response.content
    
class QuestionGeneratorAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS speaking question generator for all test parts.",
        self.instructions=[
            "Generate part-specific questions (Parts 1, 2, and 3)",
            "Ensure questions match official IELTS format",
            "Create questions suitable for 2-3 minute responses",
            "Include appropriate follow-up questions",
            "Vary topics based on test part requirements",
            "Maintain consistent difficulty level"
        ]
    
    def generate_question(self, question_number):
        topics = {
            1: "personal background and interests",
            2: "work, study, or future plans",
            3: "abstract topic requiring opinion and reasoning"
        }
        
        prompt = f"""
        Generate an IELTS Speaking Part {question_number} question about {topics[question_number]}.
        For Part {question_number}:
        
        - Make it open-ended and conversation-like
        - Suitable for 2-3 minute responses
        - Include follow-up points if needed
        - Match official IELTS speaking test style
        
        Format: Return only the question text, no additional content.
        """
        
        response = self.run(prompt)
        return response.content

class SpeechRecordingAgent:
    def __init__(self):

        self.sample_rate = 44100
        self.duration = 180  
        self.recording = None
        self.is_recording = False
    
    def start_recording(self):
        self.is_recording = True
        self.recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16
        )
    
    def stop_recording(self):
        if self.is_recording:
            sd.stop()
            self.is_recording = False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"speaking_response_{timestamp}.wav"
            
            actual_samples = len(self.recording[~np.all(self.recording == 0, axis=1)])
            self.recording = self.recording[:actual_samples]
            
            wavio.write(filename, self.recording, self.sample_rate, sampwidth=2)
            return filename
        return None

class SpeechTranscriptionAgent(BaseAgent):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.recognizer = sr.Recognizer()
        self.description="Specialized speech-to-text agent for IELTS speaking responses.",
        self.instructions=[
            "Convert audio recordings to accurate text",
            "Maintain speaker's original word choice",
            "Handle various English accents and pronunciations",
            "Process clear audio files for optimal results",
            "Report any transcription issues or errors"
        ]
        
    def transcribe_audio(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                transcript = self.recognizer.recognize_google(audio_data)
                return transcript
            except Exception as e:
                return f"Error transcribing audio: {str(e)}"

class SpeakingEvaluationAgent(BaseAgent):

    def __init__(self, name, model):
        super().__init__(name, model)
        self.description="Expert IELTS speaking evaluator providing comprehensive assessment.",
        self.instructions=[
            "Evaluate fluency and coherence (0-9 band)",
            "Assess lexical resource and vocabulary usage",
            "Analyze grammatical range and accuracy",
            "Evaluate pronunciation quality",
            "Provide detailed feedback for each criterion",
            "Calculate overall band score",
            "Suggest specific improvements for each area"
        ]
    
    def evaluate_response(self, question, transcript):
        prompt = f"""
        Evaluate this IELTS Speaking response:
        
        Question: {question}
        Response transcript: {transcript}
        
        Provide detailed scoring and feedback on:
        1. Fluency and Coherence (out of 9):
           - Natural flow of speech
           - Logical organization
           - Use of discourse markers
        
        2. Lexical Resource (out of 9):
           - Vocabulary range
           - Word choice accuracy
           - Idiomatic expressions
        
        3. Grammatical Range and Accuracy (out of 9):
           - Sentence structures
           - Tense/aspect control
           - Error frequency
        
        4. Pronunciation (out of 9):
           - Clear speech
           - Word/sentence stress
           - Natural intonation
        
        5. Overall Band Score (average of above)
        
        For each criterion:
        - Justify the score with specific examples
        - Provide improvement suggestions
        """
        
        response = self.run(prompt)
        return response.content
    


def main():

    st.set_page_config(layout="wide")

    image_path = "ZZ2.jpg"
    background_image_base64 = get_base64_encoded_image(image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;  
            background-repeat: no-repeat;
            background-attachment: fixed;  
            background-position: center center;  
            color: white;
        }}
        .chat-input {{
            background-color: rgba(255, 255, 255, 0.8);  
        }}
        .css-18e3th9 {{
            padding: 0rem; 
        }}
        .stButton>button {{
            color: #4CAF50;
            border-radius: 10px;
            border: 2px solid #4CAF50;
        }}
        .stTextInput>div>div>input {{
            color: #fffff;
        }}
        .css-1cpxqw2 {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("IELTS Evaluation System 📚")

    reading_agent = IELTSReadingAgent("Reading Agent", gemini_model)
    question_agent = QuestionGeneratorAgentReading("Question Agent", gemini_model)
    evaluation_agent = AnswerEvaluationAgent("Evaluation Agent", gemini_model)
    audio_agent = AudioProcessingAgent("Audio Agent", gemini_model)
    listening_agent = ListeningEvaluationAgent("Listening Agent", gemini_model)
    writing_prompt_agent = WritingPromptAgent("Writing Prompt Agent", gemini_model)
    writing_agent = WritingEvaluationAgent("Writing Agent", gemini_model)

    tab1, tab2, tab3, tab4 = st.tabs(["Reading", "Listening", "Writing", "Speaking"])

    with tab1:
        st.header("Reading Section")
        
        if 'passage' not in st.session_state:
            pdf_path = "IELTS.pdf"
            passage = reading_agent.process_pdf(pdf_path)
            questions = question_agent.generate_questions(passage)
            st.session_state.passage = passage
            st.session_state.questions = questions

        st.markdown("""
        <style>
        .reading-passage {
            background-color: #0000;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            line-height: 1.8;
            text-align: justify;
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.subheader("Reading Passage")
        st.markdown(
            f'<div class="reading-passage">{st.session_state.passage}</div>',
            unsafe_allow_html=True
        )
        
        st.subheader("Questions")
        user_answers = []
        for i, question in enumerate(st.session_state.questions, 1):
            st.write(f"Question {i}: {question}")
            answer = st.text_input(f"Your Answer {i}", key=f"reading_answer_{i}")
            user_answers.append(answer)

        if st.button("Submit Reading Answers"):
            evaluation = evaluation_agent.evaluate_answers(
                st.session_state.passage,
                st.session_state.questions,
                user_answers
            )
            st.write("Evaluation Results")
            st.markdown("""
            <style>
            .evaluation-result {
                background-color: #0000;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .result-header {
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .answer-section {
                margin: 15px 0;
                padding: 10px;
                border-left: 3px solid #4CAF50;
            }
            .correct {
                color: #4CAF50;
            }
            .incorrect {
                color: #f44336;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="evaluation-result">{evaluation}</div>', unsafe_allow_html=True)

    with tab2:
        st.header("Listening Section")
        
        st.markdown("""
        <style>
        .listening-evaluation {
            background-color: #0000;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .answer-section {
            margin: 15px 0;
            padding: 15px;
            background-color: #00000;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }
        .answer-header {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
        }
        .answer-detail {
            margin: 8px 0;
            padding: 4px 8px;
            line-height: 1.5;
        }
        .correct {
            color: #4CAF50;
            background-color: #E8F5E9;
        }
        .incorrect {
            color: #f44336;
            background-color: #FFEBEE;
        }
        .transcript-section {
            background-color: #E3F2FD;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        audio_path = "Ielts.mp3"
        
        if os.path.exists(audio_path):
            st.audio(audio_path)
            st.markdown("<div class='question-container'>", unsafe_allow_html=True)
            listening_answers = []
            for i in range(3):
                answer = st.text_input(
                    f"Question {i+1}",
                    key=f"listening_answer_{i}",
                    placeholder="Type your answer here..."
                )
                listening_answers.append(answer)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Submit Listening Answers", key="submit_listening"):
                with st.spinner("Evaluating answers..."):
                    transcript = audio_agent.transcribe_audio(audio_path)
                    # st.markdown("<div class='transcript-section'>", unsafe_allow_html=True)
                    # st.subheader("Audio Transcript")
                    # st.write(transcript)
                    # st.markdown("</div>", unsafe_allow_html=True)
                    
                    evaluation = listening_agent.evaluate_listening_answers(
                        transcript,
                        listening_answers
                    )
                    
                    st.markdown("<h3>Evaluation Results</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='listening-evaluation'>{evaluation}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.error("Audio file not found in the specified path")

    with tab3:
        st.header("Writing Section")
        
        if 'writing_prompt' not in st.session_state:
            st.session_state.writing_prompt = writing_prompt_agent.generate_writing_prompt()
            
        if st.button("Generate New Topic"):
            st.session_state.writing_prompt = writing_prompt_agent.generate_writing_prompt()
            
        st.markdown("""
        <style>
        .writing-prompt {
            background-color: #00000;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            white-space: pre-line;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="writing-prompt">{st.session_state.writing_prompt}</div>',
            unsafe_allow_html=True
        )

        writing_response = st.text_area("Your Response:", height=300, help="Type your essay here. Aim for at least 250 words.")
    
        if writing_response:
            word_count = len(writing_response.split())
            st.caption(f"Word count: {word_count}/250")
        
        if st.button("Evaluate Writing") and writing_response:
            evaluation = writing_agent.evaluate_writing(writing_response, st.session_state.writing_prompt)
            st.write("Evaluation Results:")
            st.markdown(evaluation)

    with tab4:
        st.title("IELTS Speaking Test System 🎙️")

        if 'agents' not in st.session_state:
            st.session_state.agents = {
                'question': QuestionGeneratorAgent("Question Generator", gemini_model),
                'recording': SpeechRecordingAgent(),
                'transcription': SpeechTranscriptionAgent("Transcription Agent", gemini_model),
                'evaluation': SpeakingEvaluationAgent("Evaluation Agent", gemini_model)
            }

        if 'speaking_initialized' not in st.session_state:
            st.session_state.current_question = 1
            st.session_state.responses = []
            st.session_state.recording_state = 'ready'
            st.session_state.current_question_text = st.session_state.agents['question'].generate_question(1)
            st.session_state.speaking_initialized = True

        st.progress(st.session_state.current_question / 3)

        if st.session_state.current_question <= 3:
            st.subheader(f"Question {st.session_state.current_question} of 3")
            st.write(st.session_state.current_question_text)

            if st.session_state.recording_state == 'ready':
                if st.button("Start Recording", key="start_recording"):
                    st.session_state.agents['recording'].start_recording()
                    st.session_state.recording_state = 'recording'
                    st.rerun()

            elif st.session_state.recording_state == 'recording':
                st.write("Recording... Speak now!")
                if st.button("Stop Recording", key="stop_recording"):
                    with st.spinner("Processing your response..."):
                        audio_file = st.session_state.agents['recording'].stop_recording()
                        transcript = st.session_state.agents['transcription'].transcribe_audio(audio_file)
                        evaluation = st.session_state.agents['evaluation'].evaluate_response(
                            st.session_state.current_question_text,
                            transcript
                        )

                        st.session_state.responses.append({
                            'question': st.session_state.current_question_text,
                            'audio_file': audio_file,
                            'transcript': transcript,
                            'evaluation': evaluation
                        })

                        st.session_state.recording_state = 'complete'
                        st.rerun()

            elif st.session_state.recording_state == 'complete':
                response = st.session_state.responses[-1]
                st.write("Your response (transcribed):", response['transcript'])
                st.markdown(response['evaluation'])

                if st.button("Continue to Next Question", key="continue"):
                    st.session_state.current_question += 1
                    if st.session_state.current_question <= 3:
                        st.session_state.current_question_text = st.session_state.agents['question'].generate_question(
                            st.session_state.current_question
                        )
                    st.session_state.recording_state = 'ready'
                    st.rerun()

        else:
            st.header("Speaking Test Complete!")

            for i, response in enumerate(st.session_state.responses, 1):
                st.subheader(f"Question {i}")
                st.write("Question:", response['question'])
                st.write("Your response (transcribed):", response['transcript'])
                st.write("Evaluation:")
                st.markdown(response['evaluation'])
                st.audio(response['audio_file'])
                st.divider()

            if st.button("Start New Test", key="new_test"):
                st.session_state.current_question = 1
                st.session_state.responses = []
                st.session_state.recording_state = 'ready'
                st.session_state.current_question_text = st.session_state.agents['question'].generate_question(1)
                st.rerun()

if __name__ == "__main__":
    main()