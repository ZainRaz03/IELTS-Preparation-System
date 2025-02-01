
# IELTS Evaluation System

The IELTS Evaluation System is a sophisticated platform designed to help users prepare for the IELTS exams by evaluating their skills in reading, listening, writing, and speaking. This system employs a multi-agent AI architecture, leveraging various technologies and frameworks to automate question generation, answer evaluation, and the transcription of spoken responses.

## Features

- **Reading Evaluation**: Automates the processing of IELTS reading passages and generates comprehension questions. Provides detailed feedback based on user responses.
- **Listening Evaluation**: Transcribes audio clips and assesses user responses, testing listening comprehension effectively.
- **Writing Evaluation**: Dynamically generates writing prompts and evaluates user responses according to IELTS scoring criteria.
- **Speaking Evaluation**: Records and transcribes user's spoken responses, then evaluates them for fluency, coherence, lexical resource, and pronunciation.

## Technologies Used

- **Streamlit**: Powers the web-based interface, facilitating an interactive user experience.
- **Python**: Serves as the backend programming language, orchestrating the data processing and AI operations.
- **PyPDF2**: Extracts text from PDF documents to use as reading passages.
- **SpeechRecognition and Pydub**: Handle audio file manipulation and transcription for the listening and speaking evaluations.
- **Sounddevice**: Captures audio directly from the user's microphone for real-time speaking evaluation.
- **ChromaDB**: Manages data persistence for storing and retrieving passages and user responses.
- **Gemini Model from Google**: Utilized for natural language processing tasks, including text generation and response evaluation.
- **Base64**: Embeds background images directly into the web application for enhanced aesthetics.
- **Phidata**: Utilized for building and managing the multi-agent architecture that underpins the entire system.

## Installation

Follow these steps to get the system up and running locally:

1. Clone the repository:
  
   git clone https://github.com/ZainRaz04/IELTS-Preparation-System.git
   
2. Navigate to the project directory:
   
   cd IELTS-Preparation-System
   
3. Install required Python packages:
   
   pip install -r requirements.txt
   
4. Run the application:

   streamlit run IELTS.py
   

## Usage

Navigate through the application via the Streamlit interface:

- **Reading Tab**: Engage with automatically generated reading comprehension questions.
- **Listening Tab**: Practice listening skills with transcribed responses.
- **Writing Tab**: Respond to IELTS Task 2 writing prompts.
- **Speaking Tab**: Test speaking abilities through recorded and evaluated responses.

## Contributing

Contributions are highly appreciated! Please fork the repository and submit a pull request with your enhancements or fixes.

