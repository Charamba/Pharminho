import openai
openai.__version__
from openai import AzureOpenAI, OpenAI
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from configparser import ConfigParser, ExtendedInterpolation

def init_config(config_filename):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_filename, 'UTF-8')
    return config

def init_LLM_client(config, http_client):
    client = AzureOpenAI(
        api_key=config['OPENAI']['OPENAI_API_KEY'],  
        api_version=config['OPENAI']['OPENAI_API_VERSION'],
        base_url=config['OPENAI']['AZURE_OPENAI_BASE_URL'],
        http_client=http_client
    )
    return client

def send_question(messages, client, engine, max_response_tokens=500):
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=0.0,
        max_tokens=max_response_tokens,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def init_embedding_client_AzureOpenAi(config, http_client, engine):
    OPENAI_API_BASE = config['OPENAI']['OPENAI_API_BASE']
    AZURE_OPENAI_PREFIX = config['OPENAI']["AZURE_OPENAI_PREFIX"]
    OPENAI_API_VERSION = config['OPENAI']['OPENAI_API_VERSION']

    APIKEY_MODELOS_IA = config['OPENAI']['OPENAI_API_KEY']#environ['APIKEY_MODELOS_IA']

    OPENAI_BASE_URL = f'{OPENAI_API_BASE}/{AZURE_OPENAI_PREFIX}'

    embedding_client = AzureOpenAIEmbeddings(
        model=engine,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=APIKEY_MODELOS_IA,
        base_url=f'{OPENAI_BASE_URL}/{engine}',
        http_client=http_client
    )

    return embedding_client

def init_embedding_client(openai_key, http_client, model_name):
    embedding_client = OpenAIEmbeddings(
        model=model_name,  # ou outro modelo suportado
        api_key=openai_key,     # ou omita se estiver usando variável de ambiente
        http_client=http_client
    )

    return embedding_client

def generate_embedding(embedding_client, text):
    return embedding_client.embed_query(text)
