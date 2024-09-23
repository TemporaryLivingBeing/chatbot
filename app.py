from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
import chromadb
from chromadb.config import Settings
from config import openai_api_key

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from random import seed
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
OPENAI_API_KEY = openai_api_key

with open('ta_data.txt', 'r', encoding='utf-8') as file:
    ta_texts = file.read()

with open('advisor_data.txt', 'r', encoding='latin-1') as file:
    advisor_texts = file.read()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def split_text_into_chunks(text, delimiter="url", max_length=7000):
    chunks = text.split(delimiter)

    chunks = [delimiter + chunk.strip() for chunk in chunks if chunk.strip()]

    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_length:
            split_point = chunk.rfind(' ', 0, max_length)
            if split_point == -1:
                split_point = max_length
            final_chunks.append(chunk[:split_point])
            chunk = chunk[split_point:].strip()
        final_chunks.append(chunk)

    return final_chunks

ta_chunks = split_text_into_chunks(ta_texts)
advisor_chunks = split_text_into_chunks(advisor_texts)

# load it into Chroma
ta_vectorstore = Chroma.from_texts(ta_chunks, embeddings)
advisor_vectorstore = Chroma.from_texts(advisor_chunks, embeddings)

ta_retriever = ta_vectorstore.as_retriever()
advisor_retriever = advisor_vectorstore.as_retriever()


llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.\
Use 200 words maximum and keep the answer concise.\
""" #gives the AI the prompt to create a new prompt using the context and og prompt, also insures that answer is less than or equal to 100 words to minimize token use
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ta_history_aware_retriever = create_history_aware_retriever( # retrieves documents using the prompt that has conversation context
    llm, ta_retriever, contextualize_q_prompt                # apparently uses 2 AI calls, one for the retriever and another for the contextualizing the prompt
)

advisor_history_aware_retriever = create_history_aware_retriever( # retrieves documents using the prompt that has conversation context
    llm, advisor_retriever, contextualize_q_prompt                # apparently uses 2 AI calls, one for the retriever and another for the contextualizing the prompt
)


ta_system_prompt = """You are a teachers assistant for CSC-272 at Furman University. \
You are to help students with coursework, test prep, and all things Data Mining. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use 200 words maximum and keep the answer concise.\
{context}""" #defines roll of AI and gives context, also insures that answer is less than or equal to 100 words to minimize token use

advisor_system_prompt = """You are an academic advisor for Furman University. \
You are to help students with questions regarding course planning, scheduling, major and minor requirments and General Education Requirements (GERs). \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use 200 words maximum and keep the answer concise.\
{context}""" #defines roll of AI and gives context, also insures that answer is less than or equal to 100 words to minimize token use


ta_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ta_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
advisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", advisor_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
ta_question_answer_chain = create_stuff_documents_chain(llm, ta_prompt) #creates the answer using history and contextualized prompt
advisor_question_answer_chain = create_stuff_documents_chain(llm, advisor_prompt) #creates the answer using history and contextualized prompt

ta_rag_chain = create_retrieval_chain(ta_history_aware_retriever, ta_question_answer_chain)
advisor_rag_chain = create_retrieval_chain(advisor_history_aware_retriever, advisor_question_answer_chain)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatTA', methods=['POST'])
def chatTA():
    user_message = request.json.get('message')
    context = request.json.get('context')
    print("Message is", user_message, "and context is", context) 
    response = useChatGPTTA(user_message, context)
    return jsonify({'message': response})


@app.route('/chatAdvisor', methods=['POST'])
def chatAdvisor():
    user_message = request.json.get('message')
    context = request.json.get('context')
    print("Message is", user_message, "and context is", context) 
    response = useChatGPTAdvisor(user_message, context)
    return jsonify({'message': response})


def useChatGPTAdvisor(user_message, history):
    try:
        context = historyMaker(history)
        print(context)
        response = advisor_rag_chain.invoke(
            {
                "input": user_message,
                "chat_history": context.messages
            }
        )
        if isinstance(response, dict):
            response = {k: str(v) for k, v in response.items()}
        else:
            response = str(response)

        return response

    except Exception as e:
        print("Error in useChatGPTTA:", e)
        return str(e)

def useChatGPTTA(user_message, history):
    try:
        context = historyMaker(history)
        print(context)
        response = ta_rag_chain.invoke(
            {
                "input": user_message,
                "chat_history": context.messages
            }
        )
        if isinstance(response, dict):
            response = {k: str(v) for k, v in response.items()}
        else:
            response = str(response)

        return response

    except Exception as e:
        print("Error in useChatGPTTA:", e)
        return str(e)

def extract_urls_from_content(content):
    urls = []
    for doc in content:
        tmp = doc.page_content
        words = tmp.split() 
        for word in words:
            if word.startswith("http://") or word.startswith("https://"):
                urls.append(word)
    return urls

def historyMaker(context):
    history = ChatMessageHistory()
    lines = context.split('\n')

    for line in lines:
        if line.startswith('AI:'):
            history.add_ai_message(line[3:].strip())
        elif line.startswith('User:'):
            history.add_user_message(line[5:].strip())

    print("History:", history)
    return history
print("\n\n\n\n\n\n\n\\n\n\n\n\n\n\n\n\\n\nThere\n\n\n\n\n\n\n\n\nHello")
if __name__ == '__main__':
    app.run(debug=True)
	
