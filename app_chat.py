import streamlit as st
from openai import OpenAI
import ssl
import httpx

diretrizes = """
# Diretrizes do Chatbot para Farmácia Escola da UFPE

Você é um assistente virtual da Farmácia Escola da UFPE. Suas respostas devem seguir rigorosamente as seguintes diretrizes:

## Escopo de Atendimento

Você pode responder apenas perguntas relacionadas a:
1. **Medicamentos disponíveis em nossa farmácia**  
    Refira-se exclusivamente aos itens listados no inventário abaixo.
2. **Operações da farmácia**  
    Tópicos incluem horários de funcionamento, serviços, entrega, localização e outros detalhes operacionais.

## Tópicos Fora do Escopo

Se um usuário perguntar sobre:
- Conselhos médicos não relacionados ao nosso inventário
- Doenças, sintomas ou tópicos gerais de farmácia
- Qualquer assunto fora do escopo acima

**Recuse educadamente** responder e redirecione o usuário para nossos canais do WhatsApp ou Instagram (veja o arquivo FAQ 2 para detalhes).

## Lista de Inventário

Discuta apenas os seguintes medicamentos e compostos:

```
Abacateiro, Alantoína, Ácido Ferúlico, Ácido Fólico, Ácido Glicirrízico, Ácido Glicólico, Ácido Hialurônico, Ácido Lático, Ácido Salicílico, Ácido Tranexamico, Ácido Tricloroacético (ATA), Ácido Kójico, Alfa Arbutin, Alfa Bisabolol, Carbonato de Cálcio, Cafeína, Captopril, Castanha da Índia, Cetoconazol, Ciclobenzaprina, Ciclopirox Olamina, Clobetasol, Cobre Quelato (Bisglicitano), Condroitina, Creatina, Cúrcuma Longa, Diacereína, Doxazosina, Dutasterida, Fosfato de Potássio, Fosfato de Sódio, Furosemida, Ginkgo Biloba (ext. seco), Glucosamina, Hidroquinona, Hidroxicloroquina, Ibuprofeno, Iodeto de Potássio, Itraconazol, L-Carnosina, LCD (licor carbonis deterg.), Magnésio Quelato (Bisglicinato), Maca Peruana, Melatonina, Minoxidil, Óleo de Amêndoas Doce, Óleo de Semente de Uva, Ômega 3, Papaína, Picolinato de cromo, Podofilina, Resorcina, Rutina, Silício (Siliciumax), Tansulosina, UC-II (Colágeno Tipo 2), Ureia, Vitamina B1 (Tiamina), Vitamina B2 (Riboflavina), Vitamina B3 (Niacinamida), Vitamina B6 (Piridoxina), Vitamina B7 (Biotina), Vitamina B12 (Cianocobalamina), Vitamina B12 (Metilcobalamina), Vitamina C (Oral e Tópica), Vitamina D3, Vitamina K2 (MK-7), Zinco Quelato (Bisglicitano)
```

---

**Nota:**  
Sempre mantenha um tom educado e profissional. Para consultas fora do escopo, utilize os canais oficiais de redirecionamento conforme especificado.
"""

def show_references(references):
    references_list = [(reference["document_name"], reference["page"]) for reference in references]
    references_dict = {}

    for doc_name, page in references_list:
        if doc_name in references_dict:
            references_dict[doc_name].append(page)
        else:
            references_dict[doc_name] = [page]

    st.subheader("Referências")

    for i, (doc_name, pages) in enumerate(references_dict.items()):
        st.write("**[" + str(i+1) + "]**")
        st.write("**Documento:** ", doc_name)

        if len(pages) > 2:
            str_pages = [p + str(", ") for p in pages[:-1]]
            str_pages = "".join(str_pages)
            str_pages = str_pages.removesuffix(", ")
            str_pages += " e " + pages[-1]
        elif len(pages) == 2:
            str_pages = pages[0] + " e " + pages[1]
        else:
            str_pages = pages[0]

        st.write("**Página(s):** ", str_pages)
        st.write("")


# Cria contexto SSL que ignora verificação
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Cria cliente HTTPX com SSL ignorado
client_httpx = httpx.Client(verify=False)

from application import get_knowledge_context
import os
# For Tiktoken
tiktoken_cache_dir = "tiktoken_cache/"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))


# Show title and description.
# st.title("💬 Chatbot")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )

st.image("pharminho-header.jpeg")#, use_column_width="always")

st.header("Pharminho")

# Ask user for their OpenAI API key via st.text_input.
# Alternatively, you can store the API key in ./.streamlit/secrets.toml and access it
# via st.secrets, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.secrets["openai_api_key"]

# openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key, http_client=client_httpx)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via st.chat_message.
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="pharminho-avatar.jpeg"):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("Faça uma pergunta"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        current_context, current_refs = get_knowledge_context(question=prompt, openai_key=openai_api_key, database_name="farmacia-ufpe")

        # messages=[
        # {"role": "system", "content": 'Você deve ajudar o usuário com a pergunta baseado no seguinte contexto: ' + current_context},
        # {"role": "user", "content": [
        #     {
        #         'type': 'text',
        #         'text': f"{prompt}"
        #     }
        # ]}
        # ]

        messages = [
        {
            "role": "system",
            "content": f"{diretrizes}\n\nBaseie-se no seguinte contexto recuperado: {current_context}"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}"
                }
            ]
        }
        ]

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages + st.session_state.messages,
            stream=True,
        )

        #text_references = write_references(current_refs)

        # Stream the response to the chat using st.write_stream, then store it in 
        # session state.
        with st.chat_message("assistant", avatar="pharminho-avatar.jpeg"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

        #st.write(text_references)
        #show_references(current_refs)

