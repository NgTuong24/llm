from langchain.chains import create_retrieval_chain
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"

CHROMA_PATH = "chroma"
DATA_PATH = "data"

embeddings = OllamaEmbeddings(model="llama3")

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

model = Ollama(model="llama3")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

retriever = db.as_retriever(search_kwargs={"k": 3})

retriever_prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

history_aware_retriever = create_history_aware_retriever(
    llm=model,
    prompt=retriever_prompt,
    retriever=retriever)

retriever_chain = create_retrieval_chain(history_aware_retriever, document_chain)


def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response


if __name__ == "__main__":
    query = input("Query: ")
    chat_history = []
    while query != "q":
        ans = process_chat(chain=retriever_chain, question=query, chat_history=chat_history)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=ans["answer"]))
        print('Answer: ', ans["answer"])
        print('Context: ')
        for docs in ans['context']:
            print(docs)
        query = input("Query: ")
