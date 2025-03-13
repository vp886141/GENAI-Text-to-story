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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

def generate_text_from_image(url: str) -> str:
    image_to_text: Any = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text: str = image_to_text(url)[0]["generated_text"]
    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text

def generate_story_from_text(scenario: str) -> str:
    prompt_template: str = f"""
    You are a talented storyteller who can create a story from a simple narrative.
    Create a story using the following scenario; the story should be a maximum of 500 words long, Fun and creative way to make story interesting;
   
    CONTEXT: {scenario}
    STORY:
    """
    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm: Any = ChatGroq(model_name="llama-3.2-11b-vision-preview", temperature=0.9)  # Using Groq's Llama model
    story_llm: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)
    generated_story: str = story_llm.predict(scenario=scenario)
    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story

def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)
    st.image("vimageeee.webp")
    with st.sidebar:
        st.image("vimageeee.webp")
        st.write("AI App created by @ PragyanAI - Education Purpose")
        st.write("Contact Sateesh Ambesange for 5 Days Workshop:pragyan.ai.school@gmail.com")
        st.write("TB - Story to Audio will be done Later")
    st.header("Image-to-Story Converter")
    uploaded_file: Any = st.file_uploader("Please choose a file to upload", type="jpg")
    if uploaded_file is not None:
        bytes_data: Any = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        progress_bar(100)
        scenario: str = generate_text_from_image(uploaded_file.name)
        story: str = generate_story_from_text(scenario)
        with st.expander("Generated Image scenario"):
            st.write(scenario)
        with st.expander("Generated short story"):
            st.write(story)

if __name__ == "__main__":
    main()
