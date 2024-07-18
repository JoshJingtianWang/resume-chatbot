import streamlit as st

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from langchain_core.runnables import chain
from typing import List, Any, Dict, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from collections import defaultdict

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

# https://python.langchain.com/v0.2/docs/how_to/add_scores_retriever/

class CustomParentDocumentRetriever(ParentDocumentRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        results = self.vectorstore.similarity_search_with_score(
            query, **self.search_kwargs
        )

        # Map doc_ids to list of sub-documents, adding scores to metadata
        id_to_doc = defaultdict(list)
        for doc, score in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                doc.metadata["score"] = score
                id_to_doc[doc_id].append(doc)

        # Fetch documents corresponding to doc_ids, retaining sub_docs in metadata
        docs = []
        for _id, sub_docs in id_to_doc.items():
            docstore_docs = self.docstore.mget([_id])
            if docstore_docs:
                if doc := docstore_docs[0]:
                    doc.metadata["sub_docs"] = sub_docs
                    docs.append(doc)

        return docs

def load_chunk_retriever(category_name):

    category_name = category_name.replace(" ", "_").lower()

    vectorstore = FAISS.load_local(f"./indices/chunks/chunk_vectorstore_{category_name}", 
                                    openai_embeddings,
                                    allow_dangerous_deserialization=True)
    
    fs = LocalFileStore(f"./indices/chunks/chunk_docstore_{category_name}")
    store = create_kv_docstore(fs)
    
    chunk_retriever = CustomParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="similarity",
        search_kwargs={'score_threshold': 0.5, 'k': 4}, # filtering for docs with similarity score < 0.5
    )
    
    return chunk_retriever

def load_summary_retriever(category_name):

    category_name = category_name.replace(" ", "_").lower()

    #print(f"Loading vector store for category: {category_name}")

    # Load the vector store
    whole_doc_summary_vectorstore = FAISS.load_local(
        f"./indices/summaries/whole_doc_summary_vectorstore_{category_name}", 
        openai_embeddings,
        allow_dangerous_deserialization=True
    )

    #print(f"Loaded vector store for {category_name}: {whole_doc_summary_vectorstore}")

    # Define the chain function within the context
    @chain
    def whole_doc_summary_retriever(query: str) -> List[Document]:
        #print(f"Executing chain function for category: {category_name} with query: {query}")
        docs, scores = zip(*whole_doc_summary_vectorstore.similarity_search_with_score(query=query, **{
                                                                                                        #'score_threshold': 0.6,  # filtering for docs with similarity score < 0.6
                                                                                                        'k' : 2
                                                                                                        }))
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        return docs

    return whole_doc_summary_retriever

# Define the categories
categories = ['IFF', 'UC Irvine', 'Roblox', 'University of Rochester', 'General']

# Initialize the retrievers dictionary
chunk_retrievers = {}
summary_retrievers = {}

# Load chunk retrievers for each category
for category in tqdm(categories):
    try:
        chunk_retrievers[category] = load_chunk_retriever(category)
    except:
        print(f"Error loading chunk retriever for {category}")
        continue

# Load the whole doc summary retrievers for each category
for category in tqdm(categories):
    try:
        summary_retrievers[category] = load_summary_retriever(category)
    except:
        print(f"Error loading summary retriever for {category}")
        continue


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

llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)

llm_to_condense = llm.bind(
    functions=condense_functions,
    function_call={"name": "Condense"}
)

# turns nested dict into a flat dict
def format_query(input_dict):
    condensed_info_dict = input_dict.pop('condensed_info')
    input_dict['condensed_history'] = condensed_info_dict['condensed_history']
    input_dict['standalone_question'] = condensed_info_dict['standalone_question']
    return input_dict

preprocess_query_chain = RunnablePassthrough.assign(condensed_info = CONDENSE_QUESTION_PROMPT                                   
                                                    | llm_to_condense 
                                                    | JsonOutputFunctionsParser()) | format_query


# ====================================================================
# Router chain and retrieval chain
# ====================================================================
class RouteQuery1(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasources: List[Literal["summary_store", "vector_store"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )

router_system_prompt1 = """You are an expert at routing a user question to the appropriate data source.
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

5. User query: "Give me a summary of Document A and explain the specifics of how mechanism B works."
    Routing: summary_store, vector_store

6. User query: "Provide an overview of the research paper on quantum computing and detailed information on the algorithms discussed."
    Routing: summary_store, vector_store
"""

router_prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", router_system_prompt1),
        ("human", "{standalone_question}"),
    ]
)

router_llm1 = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=api_key)
#router_llm1 = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
structured_llm1 = router_llm1.with_structured_output(RouteQuery1)
router_chain1 = RunnablePassthrough.assign(classification1 = (lambda x: x['standalone_question']) | router_prompt1 | structured_llm1)

class RouteQuery2(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasources: List[Literal["IFF", "UC Irvine", "Roblox", "University of Rochester", "General"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )

router_system_prompt2 = """You are an expert at routing a user question to the appropriate location based on where Josh Jingtian Wang studied or worked. 
The four possible destinations are:

1. IFF: International Flavors & Fragrances, where Josh did a data science internship.
2. UC Irvine: University of California, Irvine, where Josh did his doctorate in molecular biology.
3. Roblox: Roblox Corporation, where Josh did a data science internship.
4. University of Rochester: Where Josh did his master's in data science.
5. General: Contains Josh's resume that covers Josh's studies and work experience. Also contains other information about his life.

When deciding where to route a query, consider the following:
- Always include General as a destination to cover Josh's overall background.
- Route the query to the location most relevant to the subject of the question.
- If the query involves multiple subjects that span more than one location, route it to all relevant locations.
- If unsure, or if the query is about a specific/technical concept, route the query to all five locations to ensure comprehensive coverage.

Examples:

1. User query: "Can you tell me about Josh's data science projects?"
   Routing: University of Rochester, IFF, Roblox, General

2. User query: "What activities was Josh involved in at Roblox?"
   Routing: Roblox, General

3. User query: "Provide detailed information about Josh's studies and work experience."
   Routing: General

4. User query: "What are the key projects Josh participated in during his time at both UC Irvine and IFF?"
   Routing: UC Irvine, IFF, General

5. User query: "Can you list the key courses Josh completed that are related to artificial intelligence?"
   Routing: University of Rochester, General

6. User query: "What were Josh's primary responsibilities during his internships?"
   Routing: Roblox, IFF, General

7. User query: "How did Josh's work at IFF contribute to his academic achievements at the University of Rochester?"
   Routing: IFF, University of Rochester, General

8. User query: "Can you provide a summary of Josh's resume?"
   Routing: General

9. User query: "Tell me about Josh's professional background and skills."
   Routing: General

10. User query: "What personal interests or hobbies does Josh have outside of his professional work?"
   Routing: General

11. User query: "What is the aggregated mean awareness value for each county?"
    Routing: IFF, UC Irvine, Roblox, University of Rochester, General
"""

router_prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", router_system_prompt2),
        ("human", "{standalone_question}"),
    ]
)

router_llm2 = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
#router_llm2 = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=api_key)
structured_llm2 = router_llm2.with_structured_output(RouteQuery2)
router_chain2 = RunnablePassthrough.assign(classification2 = (lambda x: x['standalone_question']) | router_prompt2 | structured_llm2)

def remove_sub_docs(doc_dict: Dict[str, List[Document]]) -> Dict[str, List[Document]]:

    """Removes sub_docs from the context to save on LLM context window"""

    new_doc_dict = {}

    for key, docs in doc_dict.items():
        new_docs = []
        for doc in docs:
            if 'sub_docs' in doc.metadata:
                # Create a copy of the metadata without sub_docs
                new_metadata = {k: v for k, v in doc.metadata.items() if k != 'sub_docs'}
                new_doc = Document(page_content=doc.page_content, metadata=new_metadata)
                new_docs.append(new_doc)
            else:
                new_docs.append(doc)
        new_doc_dict[key] = new_docs

    return new_doc_dict

def route_query(output):

    """Route the query to the appropriate retrievers based on the classification"""

    # Extract classifications from the output
    classification1 = output.get('classification1')
    classification2 = output.get('classification2')

    # Initialize a set to store selected retrievers
    selected_retrievers = set()

    # Map the datasources to retrievers
    datasource_map = {
        'vector_store': chunk_retrievers,
        'summary_store': summary_retrievers
    }

    # Helper function to add retrievers based on classification
    def add_retrievers(location, datasources):
        for datasource in datasources:
            retriever_dict = datasource_map.get(datasource, {})
            #retriever_key = location.replace(" ", "")
            if location in retriever_dict:
                selected_retrievers.add((datasource, location))

    # Extract and combine the datasources from both classifications
    if classification1:
        datasources = classification1.datasources
    else:
        datasources = []

    if classification2:
        locations = classification2.datasources
    else:
        locations = []

    # Combine the datasources with the specific locations
    for location in locations:
        add_retrievers(location, datasources)

    #print(f"selected retrievers: {selected_retrievers}")

    # Use each selected retriever to get context
    context = {}
    standalone_question = output.get('standalone_question')
    for datasource, retriever_key in selected_retrievers:
        retriever = datasource_map[datasource][retriever_key]
        retrieved_docs = retriever.invoke(standalone_question)
        context[f"{datasource}_{retriever_key}"] = retrieved_docs

    context = remove_sub_docs(context)

    return context

retrieval_chain = RunnablePassthrough.assign(context = route_query)


# ====================================================================
# QA chain
# ====================================================================
class QAFormat(BaseModel):
    """Answer the user question/request and extract the relevant image paths from the context"""
    answer: str = Field(description="Answer to the user question/request")
    image_paths: List[str] = Field(default_factory=list, description="Relevant image paths extracted from the context. Leave empty if no relevant images are found.")

qaformat_functions = [convert_to_openai_function(QAFormat)]

qa_template = """
Answer the question/request based only on the following context and chat history. 
You can assume that Josh Jingtian Wang is involved in everything listed in the context.

You can also use the document source information from the metadata to provide additional context. 
For example, if the source is "'source': './documents/University of Rochester/time series course intro.pptx'", you can infer that it is a lecture slide from a time series course at the University of Rochester.
Use structured formatting such as bullet points or numbered lists when appropriate.

In addition, extract a maximum of 3 image paths from the context only if they are relevant to the question/request. 
The extracted image_paths should be sorted by the relevance of the documents they belong to, from most to least relevant.

**Important Instructions for Extracting Image Paths**:
1. Only extract paths that point to actual image files (e.g., .jpg, .png).
2. Ensure that the paths are valid file paths.
3. Do not extract descriptions or references to images that are not file paths.
4. Examples of valid image paths:
   - './extracted_images/University of Rochester/time series course intro/figure-4-3.jpg'
   - './extracted_images/UC Irvine/JW_Dissertation_20220824/figure-68-11.png'
5. Examples of invalid image paths:
   - 'Figure 1: A picture of Josh’s cat Emma.' (This is a description, not a file path)
   - "./documents/General/Supplementary info for Josh Jingtian Wang.pdf" (This is not an image file)

If there are no image paths in the context, leave the image_paths empty.
If there are image paths in the context but they are not relevant to the user query, leave the image_paths empty.


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

qa_llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini', openai_api_key=api_key)
#qa_llm = ChatOpenAI(temperature=0.5, model='gpt-4o', openai_api_key=api_key)
qa_llm_structured = llm.bind(
    functions=qaformat_functions,
    function_call={"name": "QAFormat"}
)

# turns nested dict into a flat dict
def format_qa_out(input_dict):
    """
    Formats the dictionary to move 'answer' and 'image_paths' to the top level.
    
    Args:
        data (dict): The input dictionary containing the nested 'initial_answer' dictionary.
    
    Returns:
        dict: The formatted dictionary with 'answer' renamed to 'initial_answer' and moved to the top level,
              along with 'image_paths'.
    """
    # Extract the nested dictionary
    initial_answer_dict = input_dict.pop('initial_answer')
    
    # Update the original dictionary with the extracted keys
    input_dict['initial_answer'] = initial_answer_dict['answer']
    input_dict['image_paths'] = initial_answer_dict['image_paths']
    
    return input_dict

qa_chain = RunnablePassthrough.assign(initial_answer = qa_prompt | qa_llm_structured | JsonOutputFunctionsParser()) | format_qa_out


# ====================================================================
# Evaluation chain
# ====================================================================
class Evaluate(BaseModel):
    """Evaluate the answer provided to a user question as satisfactory or not"""
    eval_result: str = Field(description="Evaluation result (Y or N)")

eval_functions = [convert_to_openai_function(Evaluate)]

eval_template = """You are an evaluator tasked with determining whether the answer provided to a user's question/request is satisfactory.
The answerer is an LLM that answers user queries/requests.
Consider the chat history and the user's question/request when evaluating the answer.

A satisfactory answer **must not** include responses such as "I can't answer" or "I don't have enough info." Any response that avoids these phrases or their equivalents is considered satisfactory.

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

Example 3:
User Question:
Show me a few pictures of Josh's cat.

Answer Provided:
Here are 2 pictures of Josh's cat Emma:

Evaluation: Y

Chat History:
{condensed_history}

User Question:
{question}

Answer Provided:
{initial_answer}

Answer with "Y" if the answer is satisfactory and "N" if it is not.
"""

EVAL_QUESTION_PROMPT = PromptTemplate.from_template(eval_template)

llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
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
llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
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
# def get_image_path_from_doc(context):
#     for _, docs in context.items():
#         if isinstance(docs, list):
#             for doc in docs:
#                 if hasattr(doc, 'metadata'):
#                     image_path = doc.metadata.get('image_path')
#                     if image_path and os.path.exists(image_path):
#                         return image_path

# def get_image_path_from_context(context):
#     all_docs = [
#         doc
#         for docs in context.values()
#         for doc in docs
#         if 'score' in doc.metadata
#     ] + [
#         sub_doc
#         for docs in context.values()
#         for doc in docs
#         if 'sub_docs' in doc.metadata
#         for sub_doc in doc.metadata['sub_docs']
#         if 'score' in sub_doc.metadata
#     ]

#     # Sort the documents by score
#     sorted_docs = sorted(all_docs, key=lambda d: d.metadata['score'])

#     lowest_score_doc = sorted_docs[0]
#     if 'image_path' in lowest_score_doc.metadata:
#         print(lowest_score_doc.metadata['image_path'])
#         return lowest_score_doc.metadata['image_path']
#     else:
#         print("No image path found")
#         return None

# Determine the width based on the number of images
def get_image_width(num_images):
    if num_images == 1:
        return 400
    elif num_images == 2:
        return 300
    else:
        return 200