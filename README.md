
# IELTS Evaluation System

The IELTS Evaluation System is an advanced application designed to assist in preparing for the IELTS exams by evaluating the user's abilities in reading, listening, writing, and speaking. This system integrates AI-powered agents to automate the generation of questions, evaluation of answers, and transcription of spoken responses.

## Features

- **Reading Evaluation**: Processes IELTS reading passages and generates comprehension questions. Evaluates user responses with detailed feedback.
- **Listening Evaluation**: Transcribes audio clips and evaluates user responses to listening questions.
- **Writing Evaluation**: Generates writing prompts and evaluates user-written responses based on IELTS criteria.
- **Speaking Evaluation**: Records user's spoken responses, transcribes them, and evaluates based on fluency, coherence, lexical resource, and pronunciation.

## Technologies Used

- **Streamlit**: For creating the web-based interface.
- **Python**: The backend programming language.
- **PyPDF2**: For processing PDF files to extract reading passages.
- **SpeechRecognition**: For audio transcription in the speaking evaluation.
- **Pydub and Sounddevice**: For handling audio files.
- **ChromaDB**: For managing passage data.
- **Gemini Model from Google**: For generating and evaluating text using AI.
- **Base64**: For embedding background images.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   git clone https://github.com/ZainRaz04/IELTS-Preparation-System.git

2. Navigate to the project directory:
   
   cd IELTS-Preparation-System
   
5. Install required Python packages:
   
   pip install -r requirements.txt
  
7. Run the application:

   streamlit run IELTS.py


## Usage

After launching the application, navigate through the tabs corresponding to each section of the IELTS test:

- Select the **Reading** tab to test reading comprehension.
- Choose the **Listening** tab to practice listening skills.
- Use the **Writing** tab to receive and respond to writing prompts.
- Go to the **Speaking** tab to record and evaluate spoken responses.

## Contributing

Contributions to the project are welcome! Please fork the repository and submit a pull request with your features or corrections.

.
