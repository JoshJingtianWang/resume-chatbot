import streamlit as st

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from typing import Literal, List
import os
from PIL import Image
from IPython.display import display

api_key = st.secrets["openai_key"]

# ====================================================================
# Load databases and initialize the retrievers
# ====================================================================
# Instantiate the OpenAIEmbeddings class
openai_embeddings = OpenAIEmbeddings(api_key=api_key)
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="split_parents", 
    embedding_function=openai_embeddings,
    persist_directory="./chunk_vectorstore"
)
# The storage layer for the parent documents
#store = InMemoryStore()
fs = LocalFileStore("./chunk_docstore")
store = create_kv_docstore(fs)

# Instantiate the ParentDocumentRetriever
chunk_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

whole_doc_summary_vectorstore = Chroma(embedding_function=openai_embeddings, 
                                    #collection_name="whole_doc_summary_vectorstore",
                                    persist_directory="./whole_doc_summary_vectorstore")
whole_doc_summary_retriever = whole_doc_summary_vectorstore.as_retriever()


# ====================================================================
# Preprocessing chain
# ====================================================================
class Condense(BaseModel):
    """Summarize chat history and create a standalone question/request for RAG retrieval"""
    condensed_history: str = Field(description="summarized chat history")
    standalone_question: str = Field(description="standalone question/request condensed from chat history and follow up question")

condense_functions = [convert_to_openai_function(Condense)]

condense_template = """Summarize the following chat history succinctly, 
and then combine it with the follow-up question/request to create a standalone question/request for RAG retrieval. 
Include only the information from the chat history that is necessary or useful for understanding the follow-up question/request. 
Do not include any unnecessary details or previous user questions that do not add to the context. 
Make sure the standalone question does not repeat information already answered in the chat history.

Example 1:
Chat History:
User: How do I reset my password?
Assistant: You can reset your password by clicking on 'Forgot Password' on the login page.
User: What if I don't remember my security question?

Follow Up Question: What if I don't remember my security question?

Condensed Chat History: User asked about resetting password, and the assistant provided the steps. User is now concerned about not remembering the security question.

Standalone Question for RAG Retrieval: What should I do if I don't remember my security question for resetting my password?

Example 2:
Chat History:
User: What are the benefits of using a multi-vector retriever in LangChain?
Assistant: Multi-vector retrievers allow for more accurate and efficient information retrieval by considering multiple dimensions of similarity.
User: Can it be used for large datasets?

Follow Up Question: Can it be used for large datasets?

Condensed Chat History: User inquired about the benefits of multi-vector retrievers, and the assistant explained its advantages. User now wants to know about its applicability to large datasets.

Standalone Question for RAG Retrieval: Can a multi-vector retriever in LangChain be used for large datasets?

Example 3:
Chat History:
User: How can I improve the accuracy of my language model?
Assistant: You can improve accuracy by fine-tuning your model on a more specific dataset related to your use case.
User: What if I don't have enough data for fine-tuning?

Follow Up Question: What if I don't have enough data for fine-tuning?

Condensed Chat History: User asked about improving the accuracy of their language model, and the assistant suggested fine-tuning. User now expresses concern about the lack of data for fine-tuning.

Standalone Question for RAG Retrieval: What should I do if I don't have enough data to fine-tune my language model for improved accuracy?

Now, please process the following:

Chat History:
{chat_history}

Follow Up Question/Request: {question}

Output both the condensed chat history and the standalone question/request.
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=api_key)

llm_to_condense = llm.bind(
    functions=condense_functions,
    function_call={"name": "Condense"}
)

# turns nested dict into a flat dict
def format_query(query_chain_out):
    rag_chain_input = query_chain_out['condensed_info']
    rag_chain_input['question'] = query_chain_out['question']
    rag_chain_input['system_prompt'] = query_chain_out['system_prompt']
    rag_chain_input['chat_history'] = query_chain_out['chat_history']
    return rag_chain_input

preprocess_query_chain = RunnablePassthrough.assign(condensed_info = CONDENSE_QUESTION_PROMPT                                   
                                                    | llm_to_condense 
                                                    | JsonOutputFunctionsParser()) | format_query


# ====================================================================
# Router chain and retrieval chain
# ====================================================================
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasources: List[Literal["summary_store", "vector_store"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )

router_system_prompt = """You are an expert at routing a user question to the appropriate data source.
There are two possible destinations:
1. vector_store: This store contains chunks of text from documents.
2. summary_store: This store contains summaries of documents.

When deciding where to route a query, consider the following:
- If the query asks for detailed information, specific passages, or in-depth content, route it to the Vector Store.
- If the query asks for an overview, summary, or general information about a document, route it to the Summary Store.
- If the query involves both types of information, route it to both the Vector Store and the Summary Store.

Examples:
1. User query: "What is the main idea of the document about LangChain?"
   Routing: summary_store

2. User query: "Can you provide detailed quotes and passages that illustrate the dystopian society in '1984'?"
   Routing: vector_store

3. User query: "What are the key findings of the latest climate change report?"
   Routing: summary_store

4. User query: "Can you give me specific data on the temperature changes over the past century from the climate change report?"
    Routing: vector_store

5. User query: "Provide an overview of the research paper on quantum computing and detailed information on the algorithms discussed."
    Routing: summary_store, vector_store

6. User query: "What were the non-machine learning approaches used in the study conducted by Josh to predict COVID-19 cases?"
    Routing: vector_store
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system_prompt),
        ("human", "{standalone_question}"),
    ]
)

chunk_retriever_chain = (lambda x:x['standalone_question']) | chunk_retriever
summary_retriever_chain = (lambda x:x['standalone_question']) | whole_doc_summary_retriever

def check_if_vector_store(router_output):
    classification = router_output['classification']
    if "vector_store" in classification.datasources:
        return chunk_retriever_chain
        #return "meow"
    else:
        return ""

def check_if_summary_store(router_output):
    classification = router_output['classification']
    if "summary_store" in classification.datasources:
        return summary_retriever_chain
        #return "woof"
    else:
        return ""

router_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=api_key)
#router_llm = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=api_key)
structured_llm = router_llm.with_structured_output(RouteQuery)
router_chain = RunnablePassthrough.assign(classification = (lambda x: x['standalone_question']) | router_prompt | structured_llm)
retrieval_chain = RunnablePassthrough.assign(context = {"vector_docs": check_if_vector_store, 
                                  "summary_docs": check_if_summary_store}
                                  )


# ====================================================================
# QA chain
# ====================================================================
qa_template = """Answer the question based only on the following context and chat history. 
You can assume that Josh Jingtian Wang is involved in (attended, participated, worked on, etc.) all the projects (coursework, activities, publications) listed in the context.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("user", qa_template)
])
qa_chain = RunnablePassthrough.assign(initial_answer = qa_prompt | llm | StrOutputParser())


# ====================================================================
# Evaluation chain
# ====================================================================
class Evaluate(BaseModel):
    """Evaluate the answer provided to a user question as satisfactory or not"""
    eval_result: str = Field(description="Evaluation result (Y or N)")

eval_functions = [convert_to_openai_function(Evaluate)]

eval_template = """You are an evaluator tasked with determining whether the answer provided to a user's question/request is satisfactory.
Consider the chat history and the user's question/request when evaluating the answer.
The answerer is an LLM specifically designed to answer questions about the background of Josh Jingtian Wang.

A satisfactory answer must:
1. Be related to the user's question.
2. Address the key points or concerns raised by the question.
3. Avoid responses like "I can't answer" or "I don't have enough info."

Example 1:
User Question:
What is the weather in Arlington, VA today?

Answer Provided:
I'm unable to provide real-time information like the current weather in Arlington, VA.

Evaluation: N

Example 2:
User Question:
Tell me a bit about Josh.

Answer Provided:
Josh Jingtian Wang is a researcher who has conducted experiments and studies related to biological processes, genetic constructs, and protein interactions.

Evaluation: Y

Chat History:
{condensed_history}

User Question:
{question}

Answer Provided:
{initial_answer}

Please provide a binary evaluation (Y or N) indicating whether the provided answer satisfactorily addresses the user's question based on the above criteria.

Answer with "Y" if the answer is satisfactory and "N" if it is not.
"""

EVAL_QUESTION_PROMPT = PromptTemplate.from_template(eval_template)

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=api_key)
#llm = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=api_key)

llm_to_eval = llm.bind(
    functions=eval_functions,
    function_call={"name": "Evaluate"}
)

eval_chain = RunnablePassthrough.assign(eval_result = EVAL_QUESTION_PROMPT | llm_to_eval | JsonKeyOutputFunctionsParser(key_name="eval_result"))


# ====================================================================
# External search chain
# ====================================================================
search = DuckDuckGoSearchRun()
tools = [search]
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=api_key)
functions = [convert_to_openai_function(f) for f in tools]
llm_to_search = llm.bind(functions=functions, function_call="auto")

search_template = """You are an AI assistant tasked with answering the user's question.
If you don't know the answer, you can use the DuckDuckGoSearch tool to find the information.
If the search result is not helpful, just say you don't know and encourage the user to ask questions about Josh. 
Do not hallucinate or make up information.

Chat History:
{chat_history}

User Question:
{question}
"""

SEARCH_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", search_template),
    #("placeholder", "{agent_scratchpad}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search_chain = SEARCH_QUESTION_PROMPT | llm_to_search | OpenAIFunctionsAgentOutputParser()
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | search_chain
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, max_iterations=2)

def external_search_router(eval_output):
    eval_result = eval_output['eval_result']
    if eval_result == "N":
        return agent_executor | (lambda x: x["output"])
    else:
        return eval_output['initial_answer']

external_search_chain = RunnableLambda(external_search_router)


# ====================================================================
# Complete chain
# ====================================================================
#complete_chain = preprocess_query_chain | router_chain | retrieval_chain | qa_chain


# ====================================================================
# Other functions
# ====================================================================
# Function to display images if 'image_path' is in metadata
def get_image_path_from_doc(context):
    for _, docs in context.items():
        if isinstance(docs, list):
            for doc in docs:
                if hasattr(doc, 'metadata'):
                    image_path = doc.metadata.get('image_path')
                    if image_path and os.path.exists(image_path):
                        return image_path