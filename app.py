import streamlit as st
import requests
import os
import pandas as pd
from uuid import uuid4
import sqlite3

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
#from langchain_openai import OpenAI, AzureOpenAI
#from langchain_openai import ChatOpenAI, AzureChatOpenAI
#from langchain_openai import OpenAIEmbeddings

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

from dotenv import load_dotenv


folders_to_create = ["csvs"]

for folder_name in folders_to_create:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder: {folder_name} created successfully!")
    else:
        print(f"Folder: {folder_name} already exists")

# Load API key and Initialize models
load_dotenv()
ibm_cloud_api_key = os.getenv("IBM_CLOUD_API_KEY")
project_id=os.getenv("IBM_CLOUD_PROJECT_ID")

generate_params = {
    GenParams.MAX_NEW_TOKENS : 200,
    GenParams.TEMPERATURE : 0.1,
    GenParams.DECODING_METHOD : DecodingMethods.SAMPLE
}

wml_credentials = {
                   "url": "https://us-south.ml.cloud.ibm.com",
                   "apikey":ibm_cloud_api_key
                  }

llama_70b_model_chat_model = Model(
    model_id='ibm-mistralai/mixtral-8x7b-instruct-v01-q',
    credentials=wml_credentials,
    params=generate_params,
    project_id=project_id
) 


chat_llm = WatsonxLLM(llama_70b_model_chat_model)
#
#openai_api_key = os.getenv("OPENAI_API_KEY")

#llm = OpenAI(openai_api_key=openai_api_key)
#chat_llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.4)
#embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#

def get_basic_table_details(cursor):
    cursor.execute("""
                SELECT
    m.name AS table_name, 
    p.name AS col_name,
    p.type AS col_type
    FROM sqlite_master m
    LEFT OUTER JOIN pragma_table_info((m.name)) p
    ON m.name <> p.name
    WHERE m.type = 'table'
    ORDER BY table_name;
    """)

    tables_and_columns = cursor.fetchall()
    return tables_and_columns

def save_db_details(db_uri):
    
    unique_id = str(uuid4()).replace("-","_")
    connection = sqlite3.connect(uri)
    cursor = connection.cursor()
    
    tables_and_columns = get_basic_table_details(cursor)
    
    df = pd.DataFrame(tables_and_columns, columns=['table_name', 'column_name', 'data_type'])
    # limiting data to only three tables to avoid exhausting the context limit
    df = df[df["table_name"].isin(["customers", "invoices", "invoice_items"])]
    filename_t = 'csvs/tables_' + unique_id + '.csv'
    df.to_csv(filename_t, index=False)
    
    cursor.close()
    connection.close()
    
    return unique_id



def generate_template_for_sql(query, table_info, db_uri):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(

                    f"You are an assistant that can write SQL queries."
                    f"Given the text below, write a SQL query that answers the user's question."
                    f"Only output SQL queries. Do not add any additional output. You will be penalized if you add any additional information."
                    f"DB connection string is {db_uri}"
                    f"Here is a detailed description of the table(s): "
                    f"{table_info}"
                    "Prepend and append the SQL query with three backticks '```'"
                

            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    prompt_value = template.invoke({"text":query})
    prompt_value = prompt_value.to_string()
    
    answer = chat_llm(prompt_value)
    
    return answer

def get_output_from_llm(query, unique_id, db_uri):
    ## Load the tables csv
    filename_t = 'csvs/tables_' + unique_id + '.csv'
    df = pd.read_csv(filename_t)
    
    ## For each table create a string that lists down all the columns and their datatypes
    table_info = ''
    for table in df['table_name']:
        table_info += 'Information about table' + table + ':\n'
        table_info += df[df['table_name'] == table].to_string(index=False) + '\n\n\n'
        
    return generate_template_for_sql(query, table_info, db_uri)

def execute_the_solution(solution, db_uri):
    connection = sqlite3.connect(db_uri)
    cursor = connection.cursor()
    
    _,final_query,_ = solution.split("```")
    final_query = final_query.strip('sql')
    cursor.execute(final_query)
    result = cursor.fetchall()
    return str(result)



# Function to establish connection and read metadata from the database
def connect_with_db(uri):
    st.session_state.db_uri = uri
    st.session_state.unique_id = save_db_details(uri)
    return {"message":"Connection established to database"}

def send_message(message):
    solution = get_output_from_llm(message, st.session_state.unique_id, st.session_state.db_uri)
    result = execute_the_solution(solution, st.session_state.db_uri)
    return {"message":solution + "\n\n" + "Result:\n" + result}

## Instructions

st.subheader("Instructions")
st.markdown(
    """
    1. Enter the URI of your RDS Database in the text box below.
    2. Click the **Start Chat** button to start the chat.
    3. Enter your message in the text box below and press **Enter** to send the message to the API.
    """
)

# Initialize chat history list
chat_history = []

# Input for database URI
uri = st.text_input("Enter the RDS Database URI")

if st.button("Start Chat"):
    if not uri:
        st.warning("Please enter a valid database URI")
    else:
        st.info("Connecting to the API and starting the chat")
        chat_response = connect_with_db(uri)
        if "error" in chat_response:
            st.error("Failed to start chat. Please check the URI and try again")
        else:
            st.success("Chat started successfully!")
            
# Chat with the api
st.subheader("Chat with the API")

# Initialize Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat message from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role":"user", "content":prompt})
    
    # response
    response = send_message(prompt)["message"]
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role":"assistant", "content":response})
    
# Run the streamlit app
if __name__ == "__main__":
    st.write("This is an app for starting a chat with a database")