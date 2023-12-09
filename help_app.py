import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.cassandra import Cassandra
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

import cassio

from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_ID = os.environ["ASTRA_DB_ID"]
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.

Example of your response should be:

The answer is foo
SOURCES: xyz


Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Connecting to database ...", disable_human_feedback=True)
    await msg.send()

    #
    # define Embedding model
    #
    embeddings = OpenAIEmbeddings()

    vectorstore = Cassandra(
        embedding=embeddings,
        session=None,
        keyspace=ASTRA_DB_KEYSPACE,
        table_name="helpcentre_db",
    )
    retriever = vectorstore.as_retriever()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Astra vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = "You can now ask questions!"
    await msg.send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    # activate the chain
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    # add the source documents to the answer
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            # remove all words after the | character
            title = source_doc.metadata["title"].split(" | ")[0]

            # create a Text element for the source
            text_elements.append(
                cl.Text(content=source_doc.metadata["source"], name=title)
            )

        source_names = [text_el.name for text_el in text_elements]

        # remove duplicates in source_names
        source_names = list(dict.fromkeys(source_names))

        if source_names:
            answer += "\n\nSources:"
            for source in source_names:
                answer += f"\n{source}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
