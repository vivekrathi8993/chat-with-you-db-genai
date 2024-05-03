import streamlit as st
import requests
import os
import pandas as pd
from uuid import uuid4
import sqlite3

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
#from ibm_watsonx_ai.foundation_models import Embeddings # USE WatsonxEmbeddings instead
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings

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
project_id = os.getenv("IBM_CLOUD_PROJECT_ID")
watsonx_instance_url = os.getenv("IBM_WATSONX_URL")

wml_credentials = {
                   "url": watsonx_instance_url,
                   "apikey":ibm_cloud_api_key
                  }
# Classification Model
generate_params_classify = {
    GenParams.MAX_NEW_TOKENS : 20,
    GenParams.TEMPERATURE : 0.0,
    GenParams.DECODING_METHOD : DecodingMethods.GREEDY,
    GenParams.STOP_SEQUENCES : ['yes','no'],
    GenParams.REPETITION_PENALTY : 1
}

classification_model = Model(
    model_id='google/flan-t5-xxl',
    credentials=wml_credentials,
    params=generate_params_classify,
    project_id=project_id
) 

classify_llm = WatsonxLLM(classification_model)

# Chat model
generate_params_chat = {
    GenParams.MAX_NEW_TOKENS : 100,
    GenParams.TEMPERATURE : 0.4,
    GenParams.DECODING_METHOD : DecodingMethods.SAMPLE,
    GenParams.STOP_SEQUENCES : ['<<eos>>']
}

chat_model = Model(
    model_id='ibm-mistralai/mixtral-8x7b-instruct-v01-q',
    credentials=wml_credentials,
    params=generate_params_chat,
    project_id=project_id
) 


chat_llm = WatsonxLLM(chat_model)

embeddings = WatsonxEmbeddings(
    model_id='ibm/slate-125m-english-rtrvr',
    url=wml_credentials["url"],
    apikey=wml_credentials["apikey"],
    project_id=project_id
)
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
    #df = df[df["table_name"].isin(["customers", "invoices", "invoice_items"])]
    filename_t = 'csvs/tables_' + unique_id + '.csv'
    df.to_csv(filename_t, index=False)
    
    create_vectors(filename_t, "./vectors/tables_/"+ unique_id)
    
    cursor.close()
    connection.close()
    
    return unique_id

def create_vectors(filename,persist_directory):
    loader = CSVLoader(file_path=filename, encoding='utf8')
    data = loader.load()
    vectordb = Chroma.from_documents(data, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    
def check_if_users_query_want_general_schema_information_or_sql(query):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                
                """
                f"In the text given text user is asking a question about database "
                f"Figure out whether user wants information about database schema or wants to write a SQL query"
                f"Answer 'yes' if user wants information about database schema and 'no' if user wants to write a SQL query"

                """
            ),
            HumanMessagePromptTemplate.from_template("{text}")
        ]
    )
    
    prompt_value = template.invoke({"text":query})
    prompt_value = prompt_value.to_string()
    print(prompt_value)
    answer = classify_llm(prompt_value)
    print(f"chat answer : {answer}")
    return answer


def prompt_when_user_want_general_db_information(query,db_uri):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                
                """
                f"You are an assistant who writes SQL queries."
                f"Given the text below, write a SQL query that answers the user's question."
                f"Prepend and append the SQL query with three backticks '```'"
                f"Write select query whenever possible"
                f"Connection string to this database is {db_uri}"                

                """
            ),
            HumanMessagePromptTemplate.from_template("{text}")        ]
    )
    
    answer = chat_llm(template.format_messages(text=query))
    
    prompt_value = template.invoke({"text":query})
    prompt_value = prompt_value.to_string()
    print(f"Prompt to model for Non SQL user query: {prompt_value}")
    answer = chat_llm(prompt_value)
    print(f"LLM Response when user query is not for SQL : {answer}")
    return answer


def generate_template_for_sql(query, relevant_tables, table_info):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(

                    f"You are an assistant that can write SQL queries for SQLite database."
                    f"Given the text below, write a Standard ANSI SQL query that answers the user's question."
                    f"Assume that there is/are SQL table(s) named '{relevant_tables}' "
                    f"\nHere is a detailed description of the table(s):\n "
                    f"{table_info}"
                    f"Prepend and append the SQL query with three backticks '```'"
                    f"Only return a SQLite compatible SQL query"

            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    prompt_value = template.invoke({"text":query})
    prompt_value = prompt_value.to_string()
    print(f"Prompt for SQL generation: {prompt_value}")
    answer = chat_llm(prompt_value)
    print(f"Response of SQL generation query: {answer}")
    return answer

def get_output_from_llm(query, unique_id, db_uri):
    
    # Check if user query is for sql or general database schema
    answer_to_question_general_schema = check_if_users_query_want_general_schema_information_or_sql(query=query)
    
    if answer_to_question_general_schema == 'yes':
        return prompt_when_user_want_general_db_information(query=query, db_uri = db_uri)
    
    else:
        # Load vector DB and instantiate a retriver
        vectordb = Chroma(embedding_function=embeddings, persist_directory="./vectors/tables_/"+ unique_id)
        #retriever = vectordb.as_retriever()
        
        # Search for top 5 relevant docs from vector db
        docs = vectordb.similarity_search(query,k=5)
        print(f"List of relevant documents retrieved:\n\n{docs}")
        
        relevant_tables = []
        relevant_tables_and_columns = []

        for doc in docs:
            table_name, column_name, data_type = doc.page_content.split('\n')
            print(f"Table: {table_name}, Col: {column_name}, Type: {data_type}")
            table_name=table_name.split(':')[1].strip()
            relevant_tables.append(table_name)
            column_name=column_name.split(':')[1].strip()
            data_type = data_type.split(":")[1].strip()

            relevant_tables_and_columns.append((table_name, column_name, data_type))
            
        ## Load the tables csv
        filename_t = 'csvs/tables_' + unique_id + '.csv'
        df = pd.read_csv(filename_t)
        
        # For each relevant table list down all the columns of the table
        unique_tables = []
        for table in relevant_tables:
            if table not in unique_tables:
                unique_tables.append(table)
        unique_tables_list = ', '.join(unique_tables)
                
        ## For each table create a string that lists down all the columns and their datatypes
        table_info = ''
        
        table_info = ''
        for table in unique_tables:
            table_info += f"Table name: {table}\nColumn names and column data types:\n" 
            for column,data_type in df[df['table_name'] == table][['column_name','data_type']].values:
                table_info += f"{column}, {data_type}\n"
            table_info += "\n\n"
        
        return generate_template_for_sql(query, unique_tables_list, table_info)
    


def execute_the_solution(solution, db_uri):
    
    connection = sqlite3.connect(db_uri)
    cursor = connection.cursor()
    
    _,final_query,_ = solution.split("```")
    final_query = final_query.strip('sql')
    cursor.execute(final_query)
    
    column_names = [desc[0] for desc in cursor.description]
    result = cursor.fetchall()
    df = pd.DataFrame(result, columns=column_names)
    
    print(f"SQL query output: {df}")

    return df



# Function to establish connection and read metadata from the database
def connect_with_db(uri):
    st.session_state.db_uri = uri
    st.session_state.unique_id = save_db_details(uri)
    return {"message":"Connection established to database"}

def send_message(message):
    solution = get_output_from_llm(message, st.session_state.unique_id, st.session_state.db_uri)
    #print(f"Value of Solution : {solution}")
    result = execute_the_solution(solution, st.session_state.db_uri)
    return {"message":solution,"data":result}

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
    response = send_message(prompt)
    response_text = response["message"]
    response_data = response["data"]
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
        st.dataframe(data=response_data)
    # Add assistant response to chat history
    st.session_state.messages.append({"role":"assistant", "content":response})
    
# Run the streamlit app
if __name__ == "__main__":
    st.write("This is an app for starting a chat with a database")