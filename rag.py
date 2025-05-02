from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import getpass
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['GOOGLE_API_KEY'] = "XXXXXXXXXXXX" # Ici remplacez par votre API Key !!!
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7, top_p=0.85)

def get_retriver(gemini_embeddings):

    vectorstore_disk = Chroma(
                         persist_directory="./chroma_db",
                         embedding_function=gemini_embeddings
                       )
    return vectorstore_disk.as_retriever(search_kwargs={"k": 3}) # k=3 signifie récupérer les 5 chunks les plus pertinents

def get_prompt():

    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    return PromptTemplate.from_template(llm_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm_response(question):
    retriever = get_retriver(gemini_embeddings)
    llm_prompt = get_prompt()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

def load_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def generate_vectorstore(file_path):

    pages = load_file(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunked_documents = text_splitter.split_documents(pages)

    print(f"Chargé {len(pages)} pages et créé {len(chunked_documents)} chunks.")
    vectorstore = Chroma.from_documents(
                        documents=chunked_documents,
                        embedding=gemini_embeddings,
                        persist_directory="./chroma_db"
                    )

    print(f"Base de données vectorielle créée/mise à jour dans ./chroma_db avec {len(chunked_documents)} chunks.")
