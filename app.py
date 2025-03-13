import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
#from langchain.chat_models import ChatGroq  # Using ChatGroq instead of ChatOpenAI
from groq import Groq
#from langchain.chat_models import ChatGroq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from transformers import pipeline

css_code: str = """
    <style>
    section[data-testid="stSidebar"] > div > div:nth-child(2) {
        padding-top: 0.75rem !important;
    }
   
    section.main > div {
        padding-top: 64px;
    }
    </style>
"""

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY"…
