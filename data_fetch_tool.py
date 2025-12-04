import os
import pyodbc
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langgraph.checkpoint.memory import MemorySaver
 
# --- Configuration ---
load_dotenv()
conn_str = os.getenv("CONN_STR")
openai_api_key = os.getenv("OPENAI_API_KEY")
 
if not conn_str or not openai_api_key:
    raise ValueError("Set CONN_STR and OPENAI_API_KEY in your .env file.")
 
# --- Database Connection ---
try:
    db = SQLDatabase.from_uri(
        "mssql+pyodbc:///",
        engine_args={"creator": lambda: pyodbc.connect(conn_str)},  
        include_tables = ["Customer_Demog_Detail", "Document_List_Master", "Cibil_Header", "CRM_Detail", "Control_Daily_Flag", "Rule_Master"]
    )
    print("Database connection successful.")
except Exception as e:
    print(f"Database connection failed: {e}")
    exit()
 
# --- Build Lightweight Knowledge Base ---
def build_knowledge_base(db):
    """
    Build a minimal knowledge base - just table count
    """
    try:
        tables = db.get_usable_table_names()
        return f"Database contains {len(tables)} tables with customer, document, and transaction data."
    except Exception as e:
        print(f"Error building knowledge base: {e}")
        return "Database structure available through tools."
 
# Build KB once at startup
kb_summary = build_knowledge_base(db)
 
# --- Enhanced System Prompt ---
system_prompt = f"""
You are a SQL database assistant for customer service inquiries about loan processing. Your role is to help users find information in database without exposing technical details.
 
INTERNAL KNOWLEDGE BASE (DO NOT EXPOSE TO USER):
{kb_summary}
 
CRITICAL RULES:
- NEVER list table names, column names, or SQL queries to users
- When asked about tables, respond with general descriptions only
- ALWAYS use tools to query data before responding
- ONLY answer using data from the database
- Focus on providing helpful answers to user questions
- Other then 'select' and 'insert' you cannot perform any other query.
- for insertion task dont ask re confirmation again.
 
ADAPTIVE RESPONSE FORMATTING:
- For general questions (like "how many tables"): Provide very brief 1-2 line answers
- For specific questions (like "who is prasanna devi"): Provide 2-3 line answers with key details
- For detailed questions (like "tell me everything about prasanna devi"): Provide 4-5 line answers with comprehensive information
- For very specific follow-up questions: Provide detailed answers as needed
- Don't list all database fields - only summarize key information
- Avoid technical terms and database structure details
 
HANDLING SPECIFIC QUESTIONS:
- If asked "how many tables": Respond with count only
- If asked "what tables": Respond with "The database contains information about customers, documents, credit reports, CRM details, and business rules."
- If asked about a specific person: Provide a brief summary without technical details
- If asked about documents: List document names and their purpose
- For follow-up questions: Provide additional relevant information based on previous context
- If asked about job applications, documents for jobs, or any non-loan topics: Respond with "I can only help with loan-related information from the database."

GUARDRAILS:
If the query is incomplete then ask user to provide complete information to fetch data from the database.

IMPORTANT: ADAPT RESPONSE LENGTH BASED ON QUESTION SPECIFICITY. GENERAL QUESTIONS GET BRIEF ANSWERS, SPECIFIC QUESTIONS GET DETAILED ANSWERS.
"""
 
# --- LLM and Tools Initialization ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
 
# Initialize memory BEFORE using it
memory = MemorySaver()
 
# Create agent with verbose disabled
agent = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,  # Changed to False to disable verbose output
    agent_type="tool-calling",
    checkpointer=memory,
    prefix=system_prompt
)
 
# --- Improved Pre-response Validation ---
def validate_response(response, question):
    """Check if response contains technical details that should be hidden"""
    # Only check for actual technical exposures
    technical_patterns = [
        "CREATE TABLE", "SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE", "DELETE"
    ]
    table_names = ["CRM_Detail", "Cibil_Header", "Control_Daily_Flag", "Customer_Demog_Detail", "Document_List_Master", "Rule_Master"]
    response_lower = response.lower()
    question_lower = question.lower()
    # Only flag if there are actual SQL keywords or explicit table listings
    has_sql_keywords = any(pattern.lower() in response_lower for pattern in technical_patterns)
    has_table_listings = any(table.lower() in response_lower and ("table" in response_lower and ":" in response_lower) for table in table_names)
    # Allow responses that indicate scope limitation (these are always valid)
    if "i can only help with loan-related information" in response_lower:
        return True, "Response is valid"
    # Allow responses that mention "form" or "document" in context of helping users
    if ("form" in response_lower or "document" in response_lower) and not has_sql_keywords and not has_table_listings:
        return True, "Response is valid"
    # Allow follow-up responses that don't contain technical details
    if len(question.split()) <= 3 and not has_sql_keywords and not has_table_listings:
        return True, "Response is valid"
    # Allow responses about people or general information
    if any(keyword in question_lower for keyword in ["who", "what", "when", "where", "why", "how"]) and not has_sql_keywords and not has_table_listings:
        return True, "Response is valid"
    # Flag only clear technical exposures
    if has_sql_keywords or has_table_listings:
        return False, "Response contains technical details"
    return True, "Response is valid"
 
def Database_Worker(question : str, thread_id : str='1'):
 
    try:
        print("question : ",question)
        # Use 'input' key instead of 'messages'
        result = agent.invoke(
            {"input": question},
            {"configurable": {"thread_id": thread_id}}
        )
        ai_message = result["output"]  # Changed from result["messages"][-1].content
     
        return f"\n{ai_message}"
 
    except Exception as e:
        return f"\nAn error occurred: {e}"
    
    
# while True:
#     user_input = input('you : ')
 
#     print('agent : ',Database_Worker(user_input))
 
 
 
 
 