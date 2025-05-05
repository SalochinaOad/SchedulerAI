# README for AI Appointment Scheduler Chatbot

## Project Overview
This project is an AI-powered appointment scheduling chatbot that manages user appointments. It uses OpenAI's GPT model, LangChain, and SQL databases to interact with users, process inputs, and manage appointments. The system also includes audio-to-text functionality for scheduling through speech recognition.

### Key Features
- **Natural Language Processing (NLP):** Users can input queries either by text or audio.
- **Appointment Scheduling:** Users can fetch, add, update, and delete appointments.
- **Database Integration:** The system interacts with a SQL database to manage appointment data.
- **Audio Processing:** Allows users to interact through audio messages, which are transcribed into text.

## Technologies Used
- **LangChain:** Manages language model integration, prompt handling, and agent tool execution.
- **SQLAlchemy & SQLite:** Handles database interactions.
- **OpenAI GPT:** Powers the chatbot with conversational AI capabilities.
- **Flask:** Web framework for routing and rendering the user interface.
- **SpeechRecognition (sr):** Converts audio files into text for processing.
- **dotenv:** For managing environment variables.

## Installation

### Prerequisites
- Python 3.8+
- Install required packages using `requirements.txt`
```bash
pip install -r requirements.txt
