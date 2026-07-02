from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from chunking import kas_text
from chunking import school_text

embedder = OllamaEmbeddings(model='bge-m3')
vector_db= Chroma.from_texts(texts=kas_text, embedding=embedder, persist_directory="./kapsarc_db")

embedder_2= OllamaEmbeddings(model='bge-m3')
vector_school_db=Chroma.from_texts(texts=school_text, embedding=embedder_2, persist_directory="./school_db")
