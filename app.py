import pandas as pd
import time
import openai
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from openai import OpenAI
from swarm import Swarm, Agent
from firecrawl import FirecrawlApp
import datetime
import numpy as np
import json
import seaborn as sns
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import CSVLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
# from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup
import PyPDF2

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Handy Bees AI", page_icon="", layout="wide")
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = ""


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["api_key"] = ""
    st.session_state["initial_login_state"] = False

def verify_api_key(api_key):
    try:
        # Set the OpenAI API key
        client = OpenAI(api_key=api_key)
        
        # Make a small test request to verify if the key is valid
        models = client.models.list()
        
        # If the request is successful, return True
        return True, client
    except Exception as e:
        # If there's an error, the API key is likely invalid
        return False
    
def log_in():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2 :
        st.title("Input OpenAI API Key")
    api_key = st.text_input("Input OpenAI API Key here", type="password")
    
    if st.button("Log In"):
        if verify_api_key(api_key):
            st.session_state["logged_in"] = True
            st.session_state["api_key"] = api_key
            st.session_state["initial_login_state"] = True
            st.success("Login successful!")
            
            # Use st.query_params to set the logged_in query param
            st.query_params = {"logged_in": "true"}
            st.rerun()
        else:
            st.error("Invalid credentials. Enter valid API Key.")

# Defining Handoffs and Agents # 

# df = pd.read_csv('https://raw.githubusercontent.com/gianhirakawa/ai_republic_training/refs/heads/main/parcel_dataset_20_rows.csv')

df = pd.read_excel('sales_data.xlsx', engine='openpyxl')
df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d')) # date column data type was datetime64[ns] so need to convert to object/string

def ho2_mkt_perf_bot():
    """Handoff queries to marketing performance agent."""
    st.session_state.current_agent = mkt_perf_bot
    return mkt_perf_bot

def ho2_cx_journey_bot():
    """Handoff queries to customer journey agent."""
    st.session_state.current_agent = cx_journey_bot
    return cx_journey_bot

def ho2_rev_intel_bot():
    """Handoff queries to revenue intelligence agent."""
    st.session_state.current_agent = rev_intel_bot
    return rev_intel_bot

def ho2_viz_story_bot():
    """Handoff queries to visual storyteller agent."""
    st.session_state.current_agent = viz_story_bot
    return viz_story_bot

def hob2_main_bot():
    """Handoff back queries main agent."""
    st.session_state.current_agent = main_bot
    return main_bot


main_bot = Agent(
    name = "User Helper",
    model="gpt-4o-mini",
    instructions = f"""
        Use {df} as the dataset.
        You are a user interface agent that handles all interactions with the user based on the given dataset.
        You will verify the help needed and ask clarifications regarding the inquiry and be concise.
        You may answer general questions if needed based on the dataset.
        You will handoff to other bots (Marketing Performance Agent, Customer Journey Agent or Revenue Intelligence Agent) based on the inquiry needed.
        You will ask if user was satisfied with the answer and if yes then ask to end chat.
        If user is not satisfied then continue to help.
        """,
    functions = [ho2_mkt_perf_bot, ho2_cx_journey_bot, ho2_rev_intel_bot, ho2_viz_story_bot]
)

mkt_perf_bot = Agent(
    name = "Marketing Performance Agent",
    model="gpt-4o-mini",
    instructions = f"""
        Use {df} as the dataset.
        You are an AI expert in marketing analytics and performance evaluation. Your role is to assist users in optimizing their marketing efforts by:

          - Analyzing the effectiveness of various marketing channels (e.g., email, social media, paid ads).
          - Comparing the performance of organic vs. paid campaigns across relevant metrics.
          - Tracking and calculating ROI for campaigns, considering revenue, ad spend, and other key inputs.
          - Providing actionable recommendations for budget allocation to maximize campaign impact and efficiency.

        When presenting insights:

          - Highlight trends and comparisons using data-driven evidence.
          - Offer clear, actionable suggestions to improve marketing outcomes.
          - Tailor recommendations to the user's industry and business goals.

        Assist in data interpretation and guide them on how to visualize key findings or implement changes.
        Use concise and professional language, ensuring clarity and practicality in all responses.
        Ask if user is satisfied before handoff to visual storyteller agent.
        """,
    functions = [ho2_viz_story_bot]
)

cx_journey_bot = Agent(
    name = "Customer Journey Agent",
    model="gpt-4o-mini",
    instructions = f"""
            Use {df} as the dataset.
            You are an AI specialist in customer journey mapping and user experience analysis. Your role is to assist users in understanding and improving how customers interact with their website by:

          - Mapping user paths through the site to visualize how visitors navigate.
          - Identifying conversion bottlenecks and areas where users drop off in the funnel.
          - Analyzing user behavior based on device preferences (e.g., desktop, mobile, tablet).
          - Recommending UX improvements to enhance engagement, reduce friction, and increase conversions.

        When providing insights:

          - Use visual aids (e.g., flow diagrams, heatmaps) to represent user journeys and bottlenecks when possible.
          - Offer specific, actionable suggestions tailored to the user's goals (e.g., faster load times, clearer CTAs).
          - Ensure recommendations align with industry best practices and consider both technical feasibility and user experience.

        Analyze them to produce clear and focused recommendations.
        Use concise and professional language, ensuring clarity and practicality in all responses.
        Ask if user is satisfied before handoff to visual storyteller agent.
        """,
    functions = [ho2_viz_story_bot]
)

rev_intel_bot = Agent(
    name = "Revenue Intelligence Agent",
    model="gpt-4o-mini",
    instructions = f"""
        Use {df} as the dataset.
        You are an AI expert in revenue analysis and forecasting. Your role is to help users optimize their revenue strategies by:

          - Analyzing pricing patterns to identify trends, pricing effectiveness, and opportunities for adjustments.
          - Tracking sales performance across products, regions, channels, or customer segments.
          - Identifying peak selling periods to help with inventory planning, marketing campaigns, and staffing.
          - Forecasting revenue trends based on historical data, market conditions, and seasonality.

        When providing insights:

          - Present clear and actionable recommendations for pricing strategies, sales optimization, and demand forecasting.
          - Use data visualizations (e.g., trend lines, heatmaps, bar charts) to make patterns and predictions easier to understand.
          - Ensure all outputs are tailored to the user's business context and objectives, offering explanations for trends and actionable next steps.

        Assist in their interpretation and guide them on how to implement revenue optimization strategies effectively.
        Use concise and professional language, ensuring clarity and practicality in all responses.
        Ask if user is satisfied before handoff to visual storyteller agent.
        """,
    functions = [ho2_viz_story_bot]
)

def execute_visualization(code_snippet):
    """Execute visualization code and return the figure"""
    try:
        # Create a new figure to avoid interference with other plots
        plt.figure()
        
        # Execute the code snippet in a local namespace
        local_dict = {'plt': plt, 'df': df, 'np': np}
        exec(code_snippet, globals(), local_dict)
        
        # Get the current figure
        fig = plt.gcf()
        return fig
    except Exception as e:
        return f"Error executing visualization: {str(e)}"

viz_story_bot = Agent(
    name = "Visual Storyteller Agent",
    model="gpt-4o-mini",
    instructions = f"""
        Use {df} as the dataset.
        You are an AI expert in transforming data into compelling visual narratives. Your role is to assist users in understanding and communicating data effectively by:

          - Creating data visualizations that are clear, insightful, and tailored to specific audiences.
          - Building interactive dashboards for real-time data exploration and monitoring.
          - Generating automated reports that summarize key metrics, trends, and actionable insights.
          - Presenting key findings in an engaging, professional, and easily understandable manner.
          - Build appropriate charts and graphs based on the findings of the previous agents.

        When providing visualizations:
          - Always include executable Python code blocks using ```python ``` format
          - Use matplotlib, seaborn, or other visualization libraries
          - Ensure the code is complete and can run independently
          - Include clear labels, titles, and appropriate color schemes
          - The code will be automatically executed and displayed in the chat

        When assisting users:
          - Prioritize clarity, accuracy, and relevance in your outputs.
          - Recommend best practices for visualization and dashboard design to ensure usability.
          - Use visual aids (e.g., charts, graphs, tables) to simplify complex data.
          - Tailor your responses to the user's goals, audience, and industry.

        Guide them on how to transform their data into meaningful visual outputs or provide step-by-step instructions for implementation.
        Use concise and professional language, ensuring clarity and practicality in all responses.
        Ask if further questions are needed.
        Ask if other questions based on other agents domain and if yes then handoff to other bots (Marketing Performance Agent, Customer Journey Agent, Revenue Intelligence or User Helper) based on the inquiry needed.
        Ask if user is satisfied with the answer and if yes then ask to end chat or transfer back to User Helper if needed something else.
        """,
    functions = [hob2_main_bot, execute_visualization]
)

current_agent = main_bot
# Streamlit App



def Home():
    st.session_state.current_agent = main_bot
    st.title('Handy Bees AI')
    st.write("Handy Bees AI is a powerful machine learning (ML) agent application built with OpenAI, Swarm Intelligence, Python, and Streamlit.")
    st.write("Designed as a versatile chatbot, it empowers businesses by delivering insights into marketing performance, customer journey analysis, revenue intelligence, and data visualization.")
    st.write("Its features leverage the principles of swarm intelligence for dynamic collaboration and handoffs between agents, ensuring seamless analysis and actionable insights.")    
    st.write("## Core Features of Handy Bees AI:")
    st.write("# Swarm Intelligence")
    st.write("- Collaborative Agents: Inspired by swarm behavior, Handy Bees AI deploys multiple agents that work together in a distributed and collaborative way to solve complex problems.")
    st.write("- Adaptability: Agents dynamically adapt to data changes, ensuring real-time, context-aware recommendations for marketing and revenue optimization.")
    st.write("- Scalable Analysis: Handles large-scale datasets with collective intelligence, improving the accuracy and speed of insights.")
    st.write("# Handoff Mechanisms")
    st.write("- Agent-to-Agent Handoff: Smooth transitions between agents specializing in different domains (e.g., customer journey analysis, revenue metrics).")
    st.write("- Context Retention: Ensures continuity of insights during complex multi-agent tasks, preventing data loss or redundancy.")
    st.write("# Agentic Analysis")
    st.write("- Marketing Performance Analysis - Key Performance Indicators (KPIs): Tracks metrics like CTR, conversion rates, CAC (Customer Acquisition Cost), and LTV (Lifetime Value).")
    st.write("- Customer Journey Analysis - Behavioral Insights: Identifies patterns in customer interactions and preferences at each touchpoint.")
    st.write("- Revenue Intelligence - Forecasting and Revenue Breakdown: Analyzes revenue sources, product performance, and profitability. ")
    st.write("- Data Visualization - Graphical Insights: Presents data as charts, heatmaps, and flow diagrams for better decision-making.")


def Chat():
    client = Swarm(OpenAI(api_key=st.session_state["api_key"]))
    st.title(f"Start chatting with your Handy Bees!")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_message = st.text_input("Ask anything about you want to know about the data! You can start asking about Marketing performance, Customer journey or Revenue intel related questions.")
    submit_chat_button = st.button("Submit Chat")
    
    if submit_chat_button and user_message:
        with st.spinner("Checking!"):
            struct = st.session_state.chat_history + [{"role": "user", "content": user_message}]
            response = client.run(agent=st.session_state.current_agent, messages=struct)
            
            # Check if response contains code snippet (enclosed in ```python and ```)
            content = response.messages[-1]["content"]
            if "```python" in content:
                code_blocks = content.split("```python")
                for block in code_blocks[1:]:  # Skip the first split as it's before the code
                    code = block.split("```")[0].strip()
                    # Execute the visualization code
                    fig = execute_visualization(code)
                    if isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                        plt.close()
            
            # Add the new messages to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            st.session_state.chat_history.append({"role": "assistant", "content": content})
    
    # Display full chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"{st.session_state.current_agent.name}: {message['content']}")
    
    # Add a button to exit the chat and clear history
    if st.button("End Chat"):
        st.session_state.chat_history = []
        st.rerun()

def main_page():
    with st.sidebar :
        st.image("swarm.png", use_column_width=True)
        
        with st.container() :
            l, m, r = st.columns((1, 3, 1))
            with l : st.empty()
            with m : st.empty()
            with r : st.empty()
    
        options = option_menu(
            "Dashboard", 
            ["Home", "Data", "Chat"],
            icons = ['house', 'file', 'robot', 'pin'],
            menu_icon = "book", 
            default_index = 0,
            styles = {
                "icon" : {"color" : "#dec960", "font-size" : "20px"},
                "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
                "nav-link-selected" : {"background-color" : "#262730"}
            })
    
    # Add this at the beginning of routing logic
    if 'current_page' in st.session_state:
        options = st.session_state['current_page']
        del st.session_state['current_page']  # Clear the redirect after using it

    if 'messages' not in st.session_state :
        st.session_state.messages = []

    if st.session_state.get("initial_login_state"):
        Home()
        st.session_state["initial_login_state"] = False  # Reset after redirect
        
    if 'chat_session' not in st.session_state :
        st.session_state.chat_session = None
        
    elif options == "Home" :
        Home()

    elif options == "Data" :
        Data()

    elif options == "Chat" :
        Chat()


query_params = st.query_params  # Use st.query_params for retrieval
if query_params.get("logged_in") == ["true"] or st.session_state["logged_in"]:
    main_page()
else:
    log_in()