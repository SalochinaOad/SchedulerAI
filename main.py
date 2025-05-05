from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chains import ConversationChain
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from datetime import timedelta, datetime
from flask import Flask, jsonify, request, render_template
import speech_recognition as sr
import keyboard

import sqlite3
import os
import warnings 
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_APIKEY")
output_parser = StrOutputParser()
engine = create_engine('sqlite:///appointments.db')
db = SQLDatabase(engine)
app = Flask(__name__)



chat_llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model='gpt-4o')
sql_llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, )

memory = ConversationBufferMemory()


UPLOAD_FOLDER = './audio_uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@tool
def database_information_fetcher(user_input: str) -> dict:
    """
    Fetches appointment details from the database without making changes.
    Uses the `appointments` table.

    :param user_input: The query to fetch appointment details.
    :return: The database response or error message.
    """
    print(f"Fetching database information for query: {user_input}")
    
    try:
        # Execute the SQL query
        execute_query = QuerySQLDataBaseTool(db=db)
        query_chain = create_sql_query_chain(sql_llm, db)
        sql_chain = query_chain | execute_query
        response = sql_chain.invoke({"question": user_input})
        
        print(f"Database query response: {response}")
        return response
    except Exception as e:
        print(f"Error executing query: {e}")
        return {"error": str(e)}

@tool
def database_add(user_input: str) -> dict:
    """
    Add, update, or delete records in the database.
    Uses the `appointments` table.

    :param user_input: The query to modify database records.
    :return: The database response or error message.
    """
    print(f"Handling database modification for query: {user_input}")
    try:
        # Execute the SQL query
        execute_query = QuerySQLDataBaseTool(db=db)
        query_chain = create_sql_query_chain(sql_llm, db)
        sql_chain = query_chain | execute_query
        response = sql_chain.invoke({"question": user_input})
        
        print(f"Database query response: {response}")
        return response
    except Exception as e:
        print(f"Error executing query: {e}")
        return {"error": str(e)}

@tool
def database_normalization(user_input: str) -> str:
    """
    Normalizes raw user input for structured database queries.
    Identifies key fields such as name, date, time, and action.

    :param user_input: The raw user input to be normalized.
    :return: The normalized query string.
    """
    print(f"Normalizing user input: {user_input}")
    
    # Get current dates for normalization purposes
    today = datetime.now().strftime("%d:%m:%Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d:%m:%Y")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%d:%m:%Y")
    current_datetime = datetime.now()
    normalization_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI data converter. Convert user input into a valid query format.
        - Fields: name, start time, end time, date, action (add, update, fetch, delete).
        - consider current date and time is {datetime} ,tomorrow is {tomorrow} and yesterday is {yesterday}. And calculate the remaing day to date conversion using this
        - Ignore missing fields and return structured output only for provided information.
        - Example output:
            action: add, name: John, start time: 09:00:00, end time: 10:00:00, date: 12:10:2024
            action: update, name: Alice, start time: 14:00:00, 
        '''),
        ("human", "{user_input}")
    ])
    
    chain = normalization_prompt | sql_llm
    normalized_data = chain.invoke({
        "user_input": user_input, 
        "datetime": current_datetime, 
        "tomorrow": tomorrow, 
        "yesterday": yesterday
    })
    
    print(f"Normalized data: {normalized_data}")
    return normalized_data

# Available tools for processing queries
tools = [database_information_fetcher, database_add, database_normalization]



prompt_text = """
You are a personal AI appointment scheduler chatbot designed to efficiently manage user appointments. Follow these guidelines to interact with users and manage appointments effectively:

1. **Use Database Knowledge**:
   - Answer all customer queries using the provided database. Do not hallucinate or provide information outside of the database context.

2. **Pre-Processing**:
   - Normalize user input using the `database_normalization` agent.
   - Fetch available slots or appointment details using `database_information_fetcher`.
   - Add or update appointments using `database_add`.

3. **User Interaction**:
   - Utilize conversation history to extract available names, dates, and times to avoid asking redundant questions.
   - Confirm scheduling actions with the user before finalizing any appointment.

4. **Database Query**:
   - Ensure no overlapping bookings by verifying availability.

5. **Information Validation**:
   - Engage in database actions only when all required information is provided. If any data is missing, prompt the user for the necessary details.
   - When fetching information from the database, try to use existing information to streamline the process.

6. **Verification Protocol**:
   - Verify the schedule to ensure all information is correct for booking an appointment.
   - Check that the slot is free and that the requested date and time are **not before** the current date and time (e.g., {datetime}).
   - If the requested date and time are before the current date and time, inform the user that appointments cannot be scheduled in the past and prompt them to provide a new time.
   - If the requested time range is already booked, inform the user that it is not a free slot and ask for a different time slot.

7. **Available Slots Inquiry**:
   - When a user asks for available slots, query the database for the scheduled appointments on the specified date.
   - If the date is booked, inform the user of the booked slots and provide a list of remaining available time slots on that date after the booked times.
   - If there are no bookings for that date, list all available slots after the current date and time (e.g., {datetime}).

8. **Appointment Protocol**:
   - Use `database_information_fetcher` to check the availability of the requested time.
   - Do not allow multiple appointments to be scheduled for the same time.
   - If the requested slot is already booked, provide alternative time options, ensuring the user does not receive the booked slot in the suggestions.
   - Always obtain double confirmation from the user before finalizing the appointment booking.
"""

# Constructing the prompt template with placeholders for input, history, and scratchpad
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_text), 
        ("human", "input: {input}, history: {chat_history}"), 
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# Defining the chain: prompt template -> LLM -> output parser
chain = prompt | chat_llm | output_parser

# Create agent and executor
agent = create_tool_calling_agent(chat_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, agent_scratchpad=[], verbose=False)

def record_and_transcribe_until_q(loc_audio_file_path): 
    """
    converts audio into text and returns the transcribed text 
    
    Returns:
        str: The accumulated transcription of the recorded audio.
    """
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(loc_audio_file_path) as source:
                audio_data = recognizer.record(source)
                response_text = recognizer.recognize_google(audio_data)
                return response_text
    except sr.UnknownValueError:
        return 'retry' 
    except sr.RequestError as e:
        return 'server error'

@app.route('/',  methods=['GET','POST'])
def schedulo_ai():
    if request.method=='POST':
        message = request.form.get('message')
        audio_file = request.files.get('audio')
        if audio_file:
            print('audio files')
            filename = secure_filename(audio_file.filename)
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(audio_file_path)
            user_input = record_and_transcribe_until_q(audio_file_path)
            if user_input == 'retry':
                user_input == 'Sorry for the inconvenience please try again'
            if user_input == 'server error':
                user_input == 'error in speech conversion try again or use text.'
        elif message:
            user_input = message
        else:
            return render_template('chatbot.html')
        print('final')
        current_datetime = datetime.now()
        response = agent_executor.invoke({"input": user_input, "chat_history": memory, "datetime": current_datetime})
        output = response["output"]
        # Save the conversation history
        memory.save_context({"input": user_input}, {"output": output})
        return jsonify(success=True, message=output, response=output)
    return render_template('chatbot.html')
if __name__ == "__main__":
    app.run(debug=True)