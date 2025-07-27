from models import init_config, init_LLM_client, init_embedding_client, send_question
from langchain_community.vectorstores import FAISS
import httpx
from openai import OpenAI
import ssl

from langchain_openai import OpenAIEmbeddings

# Inicializa o cliente de embeddings (você pode passar a API Key via env var OPENAI_API_KEY)


# https://platform.openai.com/docs/guides/embeddings?lang=python
# from openai import OpenAI
# client = OpenAI()

# response = client.embeddings.create(
#     input="Your text string goes here",
#     model="text-embedding-3-small"
# )

# print(response.data[0].embedding)


def chat_app_old(question, database_name, k_number=3):
    # LLM
    MODEL_DEPLOYMENT_ID = 'gpt-4o-petrobras'
    # Embedding model
    EMBEDDINGS_DEPLOYMENT_NAME='text-embedding-petrobras'

    azure_config = init_config('config-v1.x.ini')
    http_client = httpx.Client(verify='petrobras-ca-root.pem')
    llm_client = init_LLM_client(azure_config, http_client)
    embedding_client = init_embedding_client(azure_config, http_client, EMBEDDINGS_DEPLOYMENT_NAME)

    # initializing db
    biolab_db = FAISS.load_local(
        folder_path="database",
        index_name=database_name,
        embeddings=embedding_client,
        allow_dangerous_deserialization=True,
    )

    query_collection_return = biolab_db.similarity_search_with_score(query=question, k=k_number)

    context = ""
    references = []
    for document, dist in query_collection_return:
        context += document.page_content
        references.append(document.metadata)

    print("### CONTEXTO:", context)

    messages=[
    {"role": "system", "content": 'Você deve ajudar o usuário com a pergunta baseado no seguinte contexto: ' + context},
    {"role": "user", "content": [
        {
            'type': 'text',
            'text': f"{question}"
        }
    ]}
    ]

    response = send_question(messages, llm_client, engine=MODEL_DEPLOYMENT_ID, max_response_tokens=1000)

    print("############ REFs: ", references)

    return response, references

def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def chat_app(question, openai_key, database_name, k_number=3):
    # LLM
    LLM_name = "gpt-3.5-turbo"#'gpt-4o'
    # Embedding model
    EMBEDDINGS_MODEL_NAME="text-embedding-ada-002"

    # Cria contexto SSL que ignora verificação
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    http_client = httpx.Client(verify=False)
    llm_client = OpenAI(api_key=openai_key, http_client=http_client)
    embedding_client = init_embedding_client(openai_key=openai_key, http_client=http_client, model_name=EMBEDDINGS_MODEL_NAME)

    # initializing db
    vector_database = FAISS.load_local(
        folder_path="database",
        index_name=database_name,
        embeddings=embedding_client,
        allow_dangerous_deserialization=True,
    )

    query_collection_return = vector_database.similarity_search_with_score(query=question, k=k_number)

    context = ""
    references = []
    for document, dist in query_collection_return:
        context += document.page_content
        references.append(document.metadata)

    print("### CONTEXTO:", context)

    messages=[
    {"role": "system", "content": 'Você deve ajudar o usuário com a pergunta baseado no seguinte contexto: ' + context},
    {"role": "user", "content": [
        {
            'type': 'text',
            'text': f"{question}"
        }
    ]}
    ]

    response = send_question(messages, llm_client, engine=LLM_name, max_response_tokens=1000)

    print("############ REFs: ", references)

    return response, references

def get_knowledge_context(question, openai_key, database_name, k_number=3):
        # LLM
    LLM_name = "gpt-3.5-turbo"#'gpt-4o'
    # Embedding model
    EMBEDDINGS_MODEL_NAME="text-embedding-ada-002"

    # Cria contexto SSL que ignora verificação
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    http_client = httpx.Client(verify=False)
    llm_client = OpenAI(api_key=openai_key, http_client=http_client)
    embedding_client = init_embedding_client(openai_key=openai_key, http_client=http_client, model_name=EMBEDDINGS_MODEL_NAME)

    # initializing db
    vector_database = FAISS.load_local(
        folder_path="database",
        index_name=database_name,
        embeddings=embedding_client,
        allow_dangerous_deserialization=True,
    )

    query_collection_return = vector_database.similarity_search_with_score(query=question, k=k_number)

    context = ""
    references = []
    for document, dist in query_collection_return:
        context += document.page_content
        references.append(document.metadata)
    
    return context, references