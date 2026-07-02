from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama

embedder = OllamaEmbeddings(model='bge-m3')


vector_db = Chroma(
    persist_directory="kapsarc_db",
    embedding_function=embedder
)

vector_db_2 = Chroma(
    persist_directory="school_db",
    embedding_function=embedder

)

def questions(ques,the_db,):
    question = ques
    results = the_db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f""" 
    context:{context}
    question : {question}
    answer : 
    """
    response = ollama.chat(model="llama3", messages=[{"role":"user","content": prompt}])
    print(response['message']['content'])

questions("What percentage of fuel consumption has increased due to congestion? ",vector_db)
questions('what was the Materials and Methods that were used in this research',vector_db_2)

