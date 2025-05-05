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
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from datetime import datetime, timedelta
import speech_recognition as sr
import keyboard
import sqlite3
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_APIKEY")
output_parser = StrOutputParser()
engine = create_engine('sqlite:///appointments.db')
db = SQLDatabase(engine)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
memory = ConversationBufferMemory()

# Define today's, tomorrow's, and yesterday's dates in the required format
today = datetime.now().strftime("%d:%m:%Y")
tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d:%m:%Y")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%d:%m:%Y")
datetime = datetime.now()



@tool
def database_information_fetcher(user_input: str) -> dict:
    """
    Fetches appointment details from the database without making changes.
    Uses the `appointments` table.

    :param user_input: The query to fetch appointment details.
    :return: The database response or error message.
    """
    
    try:
        # Execute the SQL query
        execute_query = QuerySQLDataBaseTool(db=db)
        query_chain = create_sql_query_chain(llm, db)
        sql_chain = query_chain | execute_query
        response = sql_chain.invoke({"question": user_input})
        
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
    
    try:
        # Execute the SQL query
        execute_query = QuerySQLDataBaseTool(db=db)
        query_chain = create_sql_query_chain(llm, db)
        sql_chain = query_chain | execute_query
        response = sql_chain.invoke({"question": user_input})
        
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
    
    # Get current dates for normalization purposes
    today = datetime.now().strftime("%d:%m:%Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d:%m:%Y")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%d:%m:%Y")
    
    normalization_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI data converter. Convert user input into a valid query format.
        - Fields: name, start time, end time, date, action (add, update, fetch, delete).
        - consider today is {today} ,tomorrow is {tomorrow} and yesterday is {yesterday}. And calculate the remaing day to date conversion using this
        - Ignore missing fields and return structured output only for provided information.
        - Example output:
            action: add, name: John, start time: 09:00:00, end time: 10:00:00, date: 12:10:2024
            action: update, name: Alice, start time: 14:00:00, status: confirm
        '''),
        ("human", "{user_input}")
    ])
    
    chain = normalization_prompt | llm
    normalized_data = chain.invoke({
        "user_input": user_input, 
        "today": today, 
        "tomorrow": tomorrow, 
        "yesterday": yesterday
    })
    
    return normalized_data

# Available tools for processing queries
tools = [database_information_fetcher, database_add, database_normalization]


prompt_text = """
You are a personal AI appointment scheduler chatbot designed to efficiently manage user appointments. Follow these guidelines:

1. **rule**:
   - use the database knowledge to answer customers query.
   - do not hallucinate. 

1. **Pre-Processing**:
   - Normalize user input using the `database_normalization` agent.
   - Fetch available slots or appointment details using `database_information_fetcher`.
   - Add or update appointments using `database_add`.

2. **User Interaction**:
   - Use the conversation history to extract available names, dates, and times to avoid redundant user queries.
   - Confirm scheduling actions with the user before finalizing the appointment.

3. **Appointment Management**:
   - **Booking**: Requires name, date, start time, and end time. Mark status as 'confirm'.
   - **Cancellation**: If the user cancels an appointment by providing a start time, cancel all appointments with the same date and start time. For a range (e.g., 1 to 5), cancel all appointments within that range.
   - **Rescheduling**: Requires name, new date, start time, and end time. Mark status as 'rescheduled'.

4. **Database Query**:
   - Only book appointments if slots are available (i.e., status is not 'confirm').
   - Ensure no overlapping bookings.

5. **Information Validation**:
   - Engage database actions only when all required information is provided.
   - If data is missing, prompt the user for necessary details.

6. **Appointment Status**:
   - Inform the user if no appointments have been scheduled.
   - Always provide a clear summary or confirmation after each action.

7. **Handling Dates**:
   - Recognize and handle references to 'today', 'tomorrow', and 'yesterday' accordingly current date and time is {datetime}.

8. **Appointment Protocol**:
   - Use `database_information_fetcher` to check if the time is available.
   - Do not book multiple appointments for the same time.
   - do not schedule an meeting before {datetime}
   - If the slot is already booked, ask the user for a different time slot.
   - Get double confirmation from the user before booking the appointment.
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
chain = prompt | llm | output_parser
# Create agent and executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, agent_scratchpad=[], verbose=False)


def record_and_transcribe_until_q():
    """
    Continuously listens to the microphone and returns the transcribed text 
    after the 'q' key is pressed. Starts recording audio when the 's' key is pressed.
    
    Returns:
        str: The accumulated transcription of the recorded audio.
    """
    
    recognizer = sr.Recognizer()
    

    with sr.Microphone() as source:

        full_text = ""  
        recognizer.adjust_for_ambient_noise(source)
        print("Press `s` to speak and `q` to stop...")

        while True:
            if keyboard.is_pressed('s'):
                print("Recording...")
                audio = recognizer.listen(source)  # Record audio
                try:
                    # Transcribe the audio
                    text = recognizer.recognize_google(audio)
                    print(f"Transcribed: {text}")
                    # Accumulate the transcription
                    full_text += text + " "

                except sr.UnknownValueError:
                    print("Could not understand audio.")
                except sr.RequestError as e:
                    print(f"Error with Google Speech Recognition service; {e}")
                    
            # Check if the "q" button is pressed to stop listening
            if keyboard.is_pressed('q'):
                return full_text.strip()
                break
while True:
    user_input = record_and_transcribe_until_q()
    print('User :', user_input)
    if user_input == 'retry':
        print('Bot: Sorry for the inconvenience please try again')
        continue
    if user_input == 'exit':
        break
    response = agent_executor.invoke({"input": user_input, "chat_history": memory, "datetime": datetime})
    output = response["output"]
    print(f'Bot: {output}')
    # Save the conversation history
    memory.save_context({"input":user_input}, {"output":output})
