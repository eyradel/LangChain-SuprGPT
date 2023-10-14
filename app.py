import os
from apikey import apikey
import streamlit as st
from langchain.llms import openai, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
st.markdown(
    '<link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True,
)
st.markdown("""""", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
                header{visibility:hidden;}
                .main {
                    margin-top: -20px;
                    padding-top:10px;
                }
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    """
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #4267B2;">
    <a class="navbar-brand" href="#"  target="_blank">Suprcede GPT</a>  
    </nav>
""",
    unsafe_allow_html=True,
)
# Set the API key first
os.environ['OPENAI_API_KEY'] = apikey

st.markdown("<button style='border-radius:50px;' class='btn btn-outline-primary'>Give it a shot !</button>",unsafe_allow_html=True)

prompt = st.text_input("Plug in your prompt here")

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template='write me an info based on : {title} while we are leveraging wikipedia research:{wikipedia_research}'
)

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat history')
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True,  memory=title_memory,output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, memory=script_memory, output_key='script')
# Sequential_chain = SequentialChain(chains=[title_chain, script_chain],input_variables = ['topic'], output_variables=['title','script'], verbose=True)
wiki = WikipediaAPIWrapper()
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title,wikipedia_research=wiki_research)
    # response = Sequential_chain( {'topic': prompt})
    st.write(title)
    st.write(script)
    st.write('\n')


    with st.expander('Title History'):
        st.success(title_memory.buffer)
    with st.expander('Script History'):
        st.warning(script_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
