# !pip install langchain-community
# !pip install -U duckduckgo-search
# !pip install langchain
# !pip install langchain_google_genai
import streamlit as st
from PIL import Image
from gemini_handler_080724 import GeminiHandler

st.set_page_config(page_title="Gemini Pro with Streamlit", page_icon="♊")

# st.write("Welcome to the Gemini Pro Dashboard. You can proceed by providing your Google API Key")

# with st.expander("Provide Your Google API Key"):
#      google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")

# if not google_api_key:
#     st.info("Enter the Google API Key to continue")
#     st.stop()

# genai.configure(api_key="AIzaSyC_nyg7czp89T9I3hYsjNm5MmlEVuDeNRI")
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
import os
import vertexai
import PyPDF2

from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationResponse,
    Tool,
    grounding,
)

# setup env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'/home/srv-ati02/GIS/sxforce-backend/geminipro2-b244f332c8f6.json'

# define variables
pro_id = "geminipro2"
region = "us-west1"

# Init project
vertexai.init(project=pro_id, location=region)
#
gemini_handler = GeminiHandler()

st.title("Gemini Pro")

with st.sidebar:
    option = st.selectbox('Choose Your Model', ('gemini-1.5-pro-preview-0409', 'gemini-pro', 'gemini-pro-vision'))
    # option = st.selectbox('Choose Your Model',('gemini-pro', 'gemini-pro-vision'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = GenerativeModel(option).start_chat()
        st.session_state.model = option

    st.write("Adjust Your Parameter Here:")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    # max_token = st.number_input("Maximum Output Token", min_value=0, value =20000)
    # gen_config = genai.types.GenerationConfig(max_output_tokens=10000,temperature=temperature)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=1,
        top_k=32,
        max_output_tokens=3000,
    )
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE
    }
    st.divider()
    st.markdown("""<span ><font size=1>Connect With Me</font></span>""", unsafe_allow_html=True)

    st.divider()

    upload_image = st.file_uploader("Upload Your Image Here", accept_multiple_files=False, type=['jpg', 'png'])
    upload_text_file = st.file_uploader("Upload Your pdf Here", accept_multiple_files=True, type=['pdf'])

    if upload_image:
        image = Image.open(upload_image)
    st.divider()

    if upload_text_file:
        pdf = PyPDF2.PdfReader(upload_text_file)
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    # print(msg["content"])

if upload_image:
    if option == "gemini-1.5-pro-preview-0409":
        st.info("Please Switch to the Gemini Pro Vision")
        st.stop()
    if use_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": use_input})
        st.chat_message("user").write(use_input)
        # response=st.session_state.chat.send_message([prompt,image], generation_config = generation_config, safety_settings=safety_settings)
        response = st.session_state.chat.send_message([use_input, image], generation_config=generation_config)
        # response.resolve()
        msg = response.text

        st.session_state.chat = GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})

        st.image(image, width=300)
        st.chat_message("assistant").write(msg)

if upload_text_file:
    if option == "gemini-pro-vision":
        st.info("Please Switch to the Gemini Pro 1.5")
        st.stop()
    if use_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": use_input})
        st.chat_message("user").write(use_input)
        # response=st.session_state.chat.send_message([prompt,image], generation_config = generation_config, safety_settings=safety_settings)
        response = st.session_state.chat.send_message([use_input, pdf], generation_config=generation_config)
        # response.resolve()
        msg = response.text
        st.session_state.chat = GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})

        st.chat_message("assistant").write(msg)


else:
    if use_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": use_input})
        st.chat_message("user").write(use_input)
        # tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        # response=st.session_state.chat.send_message(prompt,generation_config = generation_config, safety_settings=safety_settings)
        # response = st.session_state.chat.send_message(use_input, generation_config = generation_config, tools=[tool])

        response = gemini_handler.run_output_pl(use_input)
        # response.resolve()
        # msg=response.text
        # print(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        output_of_analytic = gemini_handler.run_analytic()
        print("-----------Analytic-------------")
        print(output_of_analytic)
        print("-----------END Analytic-------------\n")

        print("-----------list_name_entity-------------")
        doc_get_ = response + '\n' + output_of_analytic
        list_name_entity = gemini_handler.run_get_name_entity(doc_get_)
        print(list_name_entity)
        print("----------------------------------------\n")
        response2 = gemini_handler.final_output(list_entity=list_name_entity)
        st.session_state.messages.append({"role": "assistant", "content": response2})
        st.chat_message("assistant").write(response2)
        print(response2)

# streamlit run C:\Users\nvtuong\PycharmProjects\sxforce-backend\API_Search\ui_changes_1.py