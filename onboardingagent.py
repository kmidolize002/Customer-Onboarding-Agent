# IMPORTS
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.store.memory import InMemoryStore
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from Communication_Agent.communication_agent import compiled_communication_graph
from langgraph.checkpoint.memory import InMemorySaver
import json
from data_fetch_tool import Database_Worker
from uuid import uuid4
from datetime import datetime
from typing import Optional
import os
 
 
# ============================================================================
# CONFIGURATION
# ============================================================================
 
config = {'configurable': {'thread_id': '1'}}
database_thread = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize LLM
# model = ChatOpenAI(model = 'gpt-4o-mini', api_key = "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")
model = ChatOpenAI(model = 'gpt-4o-mini', api_key = OPENAI_API_KEY)
 
 
# ============================================================================
# BACKEND SETUP - Persistent Storage for Multi-Agent State
# ============================================================================
 
def make_backend(runtime):
    """
    Composite backend for managing agent data:
    - Filesystem: Stores collected customer data
    - StoreBackend: Persistent memories for each subagent
    """
    return CompositeBackend(
        default=FilesystemBackend(
            root_dir=r"./data/onboarding_records"
        ),
        routes={
            "/memories/": StoreBackend(runtime)
        }
    )
 
 
 
# ============================================================================
# SHARED TOOLS ACROSS SUBAGENTS
# ============================================================================
 
@tool
def fetch_customer_data(query: str) -> dict:
    """
    Fetch customer information from the database and insert data into database(requires all data that needs to be inserted.)
    tool input:
        query : takes input a properly instructed natural language query as input, explicitly mentoning the task that needs to be performed.
    tool output:
        relevant data fetched according to user query if it exists.
    """
    try:
            print(query)
            return Database_Worker(query)
       
    except Exception as e:
        return {"status": "error", "message": str(e)}
 
 
@tool
def check_api_compliance(customer_id: str) -> dict:
    """
    Extract compliance data from external API endpoint.
    Returns Bureau data, KYC verification status and Bureau score.
    """
    try:
            with open('demo_bureau.json','r') as w:
                data = w.read()
            return {
                "status": "success",
                "message": f"{data}"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
 
 
# @tool
# def check_credit_status(customer_id: str) -> dict:
#     """
#     Extract credit information from external API endpoint.
#     Returns credit score and customer tier.
#     """
#     try:
#         if customer_id in API_ENDPOINTS.get("credit_check", {}):
#             return {
#                 "status": "success",
#                 "credit_data": API_ENDPOINTS["credit_check"][customer_id],
#                 "customer_id": customer_id
#             }
#         else:
#             return {
#                 "status": "error",
#                 "message": f"No credit data for customer {customer_id}"
#             }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
 
 
 
@tool
def send_onboarding_email(email_id : str, subject : str, body : str) -> dict:
    """
    for sending mail to given mail id with given subject and body.
    Types: welcome, verification_pending, ready_to_start, integration_guide
    """
    result = compiled_communication_graph.invoke({'input' : 'email id : '+email_id + ', subject : ' + subject+', body : ' + body })
    try:
        return{'status':'success', 'message': result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
 
 
# ============================================================================
# SUBAGENT 1: COMMUNICATION AGENT
# ============================================================================
 
communication_agent_prompt = """You are the Communication Agent in the Customer Onboarding System.
 
Your role is to:
1. Send welcome emails when customers join
2. Send verification emails when compliance checks are pending
3. Send integration guides when customer is ready to start
4. Provide status updates to customers
 
When the supervisor gives you a customer_id and action type, use the send_onboarding_email tool
to send appropriate emails with relevant information.
 
Always be professional, friendly, and include clear next steps in all communications.
Reference specific details about their account and plan when available."""
 
 
# ============================================================================
# SUBAGENT 2: DATA AGENT
# ============================================================================
 
data_agent_prompt = """You are the Data Agent in the Customer Onboarding System.
 
Your role is to:
1. Retrieve customer information from the database using fetch_customer_data.
2. Fetch credit and compliance information using API endpoints.
3. Organize and structure all retrieved data in a clear format.
4. Provide data summaries for the supervisor.
5. insert data into database.

IMPORTANT : never call data tool if what to fetch is not specified.

When asked about a customer, fetch ALL relevant data and organize it comprehensively.
Present data in a structured format that other agents can easily understand and process.
When asked to insert data into database then insert data into database using available tool.

**IMPORTANT**: always create a well formated """


# ============================================================================
# SUBAGENT 3: EXTRACTION AGENT
# ============================================================================

extraction_agent_prompt = """You are the Extraction Agent in the Customer Onboarding System.

Your role is to:
1. Call API endpoints to extract compliance data (check_api_compliance).
2. Call API endpoints to extract credit information (check_credit_status).
3. Call API endpoints to extract integration readiness (check_integration_status).
4. Summarize extraction results and identify any issues or blockers

Focus on extracting external data that feeds into decision-making.
Flag any failed checks or pending verifications that need supervisor attention.
Provide clear status reports for downstream processing."""


# ============================================================================
# DEFINE SUBAGENTS CONFIGURATION
# ============================================================================

subagents = [
    {
        "name": "Communication Agent",
        "description": "Handles all customer communications - sends welcome emails, verification emails, consent mail and a update mail that the customer is onboarded to higher ups",
        "system_prompt": communication_agent_prompt,
        "tools": [send_onboarding_email],  # Only this agent sends emails
        "model": model
    },
    {
        "name": "Data Agent",
        "description": "used to retrieve and insert data into database and organizes information.(all data needs to be pass which should be inserted)",
        "system_prompt": data_agent_prompt,
        "tools": [fetch_customer_data],  # Primary tool for data retrieval
        "model": model
    },
    {
        "name": "Extraction Agent",
        "description": "Extracts data from bureau for existing customer to check loan eligiblity",
        "system_prompt": extraction_agent_prompt,
        "tools": [check_api_compliance],  # API extraction
        "model": model
    }
]

# ============================================================================
# SUPERVISOR AGENT
# ============================================================================

supervisor_system_prompt = """You are  Onboarder, the Master Supervisor Agent for Customer Onboarding.

Your role is to:
1. COMMUNICATE with the CRE(Customer Representation Executive) about their onboarding status
2. DELEGATE tasks to subagents in a logical sequence
3. COORDINATE the complete onboarding workflow
4. SYNTHESIZE information from all subagents into a comprehensive onboarding report

ONBOARDING WORKFLOW SEQUENCE:
When a CRE initiates onboarding:
1. Ask CRE for PAN number of the user.
2. Check in database if user with this PAN card exists.
    -if yes then it is an existing customer.
    -else it is a new customer

3. if existing customer then fetch its on going loan or last loan from database(customers DPD,OVERDUE AMOUNT, OUTSTANDING PAUSE, CIBIL SCORE of last loan taken)
    - if customer is having a on going loan then explain him that new loan cannot
    - After showing details ask if customer wants new loan or top-up on existing loan.
        -if yes then fetch available loans and according to his cibil score tell him for which loan he is eligible.
        -if no then tell him you cannot help for anything other then onboarding

4. if new customer display him all the loan offers available for new customer's from database.
    - When the user selects a scheme category (e.g., "EB" or "NB") but multiple offers exist under that category:
        - Ask the user to select the specific offer by number or details.
        - Do NOT proceed until the specific offer is selected.
    -once he select one remember the choice : ask him for customers name, dob, address, pin code, gender, email id.
    - re-confirm the information again.
    -**IMPORTANT**:onboard the customer(insert all information(Pan card number, name, age, loan selected, address, pincode, city, gender , mail id) in database using data agent.).
    -once all information is collected send a mail to the given mail id and greet him for connecting with xyz bank and the CRE will get to you soon.
Guardrails:
- if some unexpected input is recieved from CRE then clarify with him what he wants.
- never talk outside the scope.
- if CRE distracts the flow to something outside the domain then request him politely to stick to the process.
- don't ask for extra information then the one's needed.
- if CRE stops the process of onboarding then ask him if he wants to continue, if no stop and start a new conversation, else keep him on track.

Always wait for subagent responses before moving to next step.
Be professional, clear, and ensure customers feel supported throughout their journey."""


# 3. if existing customer then fetch the customer’s bureau data using Extraction Agent 
#     -If any dues, delays, or negative bureau findings are present then inform the customer that they are not eligible for any new loan or top-up.
#     PERFORM the following VALIDATION CHECKS:

#    **A. CHECK FOR NEGATIVE FINDINGS**
#    - IF any dues, delays, write-offs, settlements, suit-filed cases, or negative bureau remarks are present → 
#        INFORM the customer they are NOT eligible for any new loan or top-up.

#    **B. CHECK CIBIL SCORE**
#    - CIBIL score MUST be **greater than 650**.
#      - IF score ≤ 650 → customer is NOT eligible.

#    **C. CHECK BUREAU REPORT FRESHNESS**
#    - Bureau report MUST be **max 7 days old**.
#      - IF older → REQUEST the CRE to fetch a fresh bureau report.

#    **D. PRODUCT-SPECIFIC DPD CHECKS**
#    - IF customer is applying for a **Personal Loan (PL)**:
#        - DPD must be **clear for 3 months and maximum 6 months**.
#        - IF DPD criteria fail → NOT eligible.

#    - IF customer is applying for a **Business Loan (BL)**:
#        - DPD must be **clear for minimum 9 months and maximum 12 months**.
#        - IF DPD criteria fail → NOT eligible.
#     -If the bureau details is clear then ask the customer whether they want a new loan or a top-up on the existing loan.
#     -If the customer says yes then fetch the available loan options and based on the customer’s CIBIL score tell them which loans they are eligible for.
#         :If the customer asks for a loan they are eligible for then check in the database what documents are required and ask the customer to provide any missing documents.
#         Also if it is an existing customer and some data is already present in the database then ask only for the missing data.
#     -If no then ask if anything else i can help with
    
# ============================================================================
# SUPERVISOR INITIALIZATION
# ============================================================================

supervisor = create_deep_agent(
    model=model,
    system_prompt=supervisor_system_prompt,
    subagents=subagents,
    # debug=True,
    store=InMemoryStore(),  # Required for StoreBackend
    backend=make_backend,
    checkpointer=InMemorySaver()  # Enable persistence
)

# ============================================================================
# EXECUTION & TEST CASES
# ============================================================================

while True:

    query = input('you : ')
    if query.lower() == 'exit':
        break
    
    result1 = supervisor.invoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    }, config)

    print('onboarder : ',result1['messages'][-1].content)
    print('='*50)

for res in result1['messages']:
    res.pretty_print()
