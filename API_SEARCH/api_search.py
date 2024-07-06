# !pip install langchain-community
# !pip install -U duckduckgo-search
# !pip install langchain
# !pip install langchain_google_genai

import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAk2SGsbPm5H-6K-rNgnIhQsBYwkm2GHhE'
genai.configure(api_key='AIzaSyAk2SGsbPm5H-6K-rNgnIhQsBYwkm2GHhE')

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
# wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=10)
wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from typing import List
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful AI assistant. Given a user question and some DuckDuckGoSearchResults article snippets, "
            "answer the user question. If none of the articles answer the question, just say you don't know."
            "\n\nHere are the context:{context}",
        ),
        ("human", "{question}"),
    ]
)

model = GoogleGenerativeAI(model="models/gemini-1.5-pro-001", temperature=0.1)

answer = prompt | model | StrOutputParser()


def format_docs_to_links(docs: str):
    key = ['snippet:', 'title:', 'link:']
    docs = docs[1:-1]
    list_ = docs.split("],")
    links = []
    for doc in list_:
      ind_3 = doc.find(key[2])
      links.append(doc[ind_3+6:])
    return links


def format_docs_to_document(docs: str):
    print("format_docs_to_document")
    key = ['snippet:', 'title:', 'link:']
    docs = docs[1:-1]
    list_ = docs.split("],")
    links = []
    documents = ""
    for doc in list_:
        ind_1 = doc.find(key[0])
        ind_2 = doc.find(key[1])
        ind_3 = doc.find(key[2])
        documents += doc[ind_1: ind_2]
    print(documents)
    return documents


def format_sources(docs: str):
    key = ['snippet:', 'title:', 'link:']
    docs = docs[1:-1]
    list_doc = docs.split("],")
    source = []
    for doc in list_doc:
        metadata = {}
        ind_2 = doc.find(key[1])
        ind_3 = doc.find(key[2])
        metadata['title'] = doc[ind_2 + 7:ind_3]
        metadata['source'] = doc[ind_3 + 6:]
        source.append(metadata)
    return source


format_out_docs = itemgetter("docs") | RunnableLambda(format_docs_to_document)

format_source = itemgetter("docs") | RunnableLambda(format_sources)

answer = prompt | model | StrOutputParser()

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=search)
        .assign(context=format_out_docs)      # truyền output key vao assign
        .assign(answer=answer)
        .assign(metatdata=format_source)
        .pick(["answer", "metatdata"])        # show output key
)

# chain.invoke("How fast are cheetahs?")
# result3 = chain.invoke("who is Obama?")
result = chain.invoke("How fast are cheetahs?")

result
