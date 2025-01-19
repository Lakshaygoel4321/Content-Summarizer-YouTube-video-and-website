from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import UnstructuredURLLoader,YoutubeLoader
import streamlit as st
import validators
import os


st.set_page_config(page_title='🦜Langchain: Text Summarization from the URL',page_icon="🦜")
st.title("🦜Langchain: Summarize the Text")
st.subheader("Provide the any Web and Youtube URL to Summarize🔥")
st.write("There are some Requisite👇")
st.write("1. If provide the youtube url so youtube video should be english language and check sub title option is available in the video")
st.write("2. Youtube videos should always have content with in minutes, not hours, because of the model's token length limitation.")

st.secrets["HF_TOKEN"]

#os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

#llm = ChatGroq(model='Gemma2-7b-It',api_key=api_key)

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.8,token=os.getenv('HF_TOKEN'))

prompt_template = """

    Provide summary for the following content in 300 word
    content:{text}
"""

prompt = PromptTemplate(
    input_variables=['text'],
    template=prompt_template
)

generic_input = st.text_input("Provide the URL",value="",label_visibility="collapsed")

header ={ 
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"

}

if st.button('Summarize the context'):

    if not generic_input.strip():
        st.info('Please Provide the URl')
        st.stop()
    
    
    elif not validators.url(generic_input):
        st.info('Provide the Correct URL:')
        st.stop()


    else:

        try:
                
            with st.spinner("waiting....."):
                
                if 'youtube.com' in generic_input:
                    loader = YoutubeLoader.from_youtube_url(generic_input,add_video_info=False)

                else:
                    loader = UnstructuredURLLoader(urls=[generic_input],ssl_verify=False,
                                            headers=header)
                        

                docs = loader.load()
                    
                chain = load_summarize_chain(llm=llm,
                                    chain_type='stuff',
                                    prompt=prompt)
                    
                response = chain.run(docs)
                st.success(response)

        except Exception as e:
            st.exception(e)

