# ====================================================================
# Imports
# ====================================================================
import streamlit as st
import time
#from system_prompt import system_prompt
from rag_utils import (preprocess_query_chain, 
                        router_chain1, 
                        router_chain2,
                        retrieval_chain, 
                        qa_chain, 
                        eval_chain, 
                        external_search_chain, 
                        get_image_width
                        )


# ====================================================================
# Setting up the streamlit page
# ====================================================================
st.set_page_config(page_title="🦜💬 ResumeChat")
# Sidebar content
st.sidebar.title("🦜💬 ResumeChat")
st.sidebar.markdown("<h4 style='margin-top: -10px; color: gray;'>Powered by LangChain, gpt-4o-mini, and gpt-3.5-turbo.</h4>", unsafe_allow_html=True)
st.sidebar.write("Discover more about Josh Jingtian Wang with this interactive AI chatbot.")
#st.sidebar.write("Based on OpenAI's gpt-3.5-turbo.")
st.sidebar.markdown("### Contact Info")
# Example contact info with icons
st.sidebar.markdown("""
- 📧 [Email](mailto:13jtjoshua@gmail.com)
- 🔗 [LinkedIn](https://www.linkedin.com/in/joshjingtianwang/)
- 🐱 [GitHub Repo](https://github.com/JoshJingtianWang/resume-chatbot/)
- 📖 [Article](https://medium.com/@joshjtw/building-an-advanced-langchain-rag-chatbot-with-image-retrieval-and-agentic-routing-519f7765aa82)
""")


# ====================================================================
# Streamlit utils
# ====================================================================
# Streamed response emulator
def text_generator(text):
    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.markdown(result, unsafe_allow_html=True)
    return container

def get_stream(text):
    for chunk in text.split():
        chunk = f"<span style='margin-right: 5px; color: gray;'>{chunk}</span>"
        time.sleep(0.03)
        yield chunk

def display_progress_message(text):
    container = write_stream(stream=get_stream(text))
    return container

def send_button_ques(question):
    """Feeds the button question to the agent for execution.
    Args:
    - question: The text of the button
    Returns: None
    """
    st.session_state.disabled = True
    st.session_state['button_question'] = question

# ====================================================================
# Streamlit chat
# ====================================================================
system_prompt = """You are a friendly AI assistant that can help answer questions about Josh Jingtian Wang. 
                    Answer with examples and verbose details to provide a comprehensive response."""

questions = [
    'What did Josh do for his internship at IFF?',
    'Show me two figures that Josh has created for his data science projects and describe them.',
    'Give me a brief overview of Josh\'s PhD dissertation.',
    'Show me a picture of Josh\'s cat laying down.',
    'What is the weather like in Arlington, VA today?'
]

# Initialize or load the conversation history
if "messages" not in st.session_state:
    first_message = 'What would you like to know about Josh?'
    st.session_state.messages = [{"role": "assistant", "content": first_message}]

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if "disabled" not in st.session_state:
    st.session_state.disabled = False  # Initialize disabled state
    buttons = st.container(border=True)
    for q in questions:
        button_ques = buttons.button(
            label=q, on_click=send_button_ques, args=[q],
            disabled=st.session_state.disabled
        )

# Handling user input and generating a response
if user_input := (st.chat_input("You:", key="user_input") or st.session_state.get("button_question")):

    # Initialize or clear progress placeholders
    if "progress_placeholders" not in st.session_state:
        st.session_state.progress_placeholders = []
    else:
        # Clear previous progress placeholders
        for placeholder in st.session_state.progress_placeholders:
            placeholder.empty()
        st.session_state.progress_placeholders.clear()
    
    # Initialize or clear assistant response placeholder
    # if "assistant_response_placeholder" not in st.session_state:
    #     st.session_state.assistant_response_placeholder = st.empty()
    # else:
    #     st.session_state.assistant_response_placeholder.empty()
    if "assistant_response_placeholder" in st.session_state:
        st.session_state.assistant_response_placeholder.empty()

    # Append user input to messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display the user input
    with st.chat_message("user"):
        st.write(user_input)

    # Get the chat history
    chat_history = st.session_state.messages
    
    # Prepare the input for the chain
    chain_input = {"system_prompt": system_prompt, "chat_history": chat_history, "question": user_input}

    # Generate and display the response with a spinner to indicate processing
    with st.spinner("Thinking..."):
        
        # Report progress at every stage of the chain

        preprocess_output = preprocess_query_chain.invoke(chain_input)
        #preprocess_progress_placeholder.write_stream(text_generator("Condensed chat history and incorporated into user query."))
        preprocess_progress_placeholder = display_progress_message("Condensed chat history and incorporated into user query.")
        st.session_state.progress_placeholders.append(preprocess_progress_placeholder)
        #st.write(f"standalone question: {preprocess_output['standalone_question']}")

        router_output1 = router_chain1.invoke(preprocess_output)
        classification1 = router_output1['classification1'].datasources
        #router_progress_placeholder.write_stream(text_generator(f"Agent classified query as: {classification}"))
        router_progress_placeholder1 = display_progress_message(f"Agent classified query as: {classification1}")
        st.session_state.progress_placeholders.append(router_progress_placeholder1)

        router_output2 = router_chain2.invoke(router_output1)
        classification2 = router_output2['classification2'].datasources
        #router_progress_placeholder.write_stream(text_generator(f"Agent classified query as: {classification}"))
        router_progress_placeholder2 = display_progress_message(f"Agent classified query as: {classification2}")
        st.session_state.progress_placeholders.append(router_progress_placeholder2)

        retrieval_output = retrieval_chain.invoke(router_output2)
        #retrieval_progress_placeholder.write_stream(text_generator("Retrieved relevant documents."))
        retrieval_progress_placeholder = display_progress_message("Retrieved relevant documents. Generating initial answer. This may take up to 5 seconds...")
        st.session_state.progress_placeholders.append(retrieval_progress_placeholder)

        #st.write(f"context: {retrieval_output['context']}")

        qa_output = qa_chain.invoke(retrieval_output)
        #qa_progress_placeholder.write_stream(text_generator("Initial answer generated."))
        qa_progress_placeholder = display_progress_message("Initial answer generated.")
        st.session_state.progress_placeholders.append(qa_progress_placeholder)

        #st.write(f"initial answer: {qa_output['initial_answer']}")

        eval_output = eval_chain.invoke(qa_output)
        eval_result = eval_output['eval_result']
        #st.write(f"eval_result: {eval_result}")
        if eval_result == "Y":
            #eval_progress_placeholder.write_stream(text_generator("Answer evaluated as satisfactory."))
            eval_progress_placeholder = display_progress_message("Answer evaluated as satisfactory.")
        else:
            #eval_progress_placeholder.write_stream(text_generator("Answer evaluated as unsatisfactory. External web search initiated.")
            
            eval_progress_placeholder = display_progress_message("Answer evaluated as unsatisfactory. External web search initiated.")
        st.session_state.progress_placeholders.append(eval_progress_placeholder)

        assistant_response = external_search_chain.invoke(eval_output)

        # Append response to the conversation history    
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # # Display the latest response
        # with st.chat_message("assistant"):
        #     st.write(assistant_response)
        
        # Display the latest response in a dedicated placeholder
        st.session_state.assistant_response_placeholder = st.empty()
        with st.session_state.assistant_response_placeholder.container():
            with st.chat_message("assistant"):
                # image_path = get_image_path_from_context(retrieval_output['context'])
                # if image_path:
                #     st.image(image_path, caption="Here is an image related to the response")
                # summary_docs = retrieval_output['context']["summary_docs"]
                st.write(assistant_response)
                if eval_result == "Y":
                    image_paths = qa_output['image_paths'][:3] # display no more than 3 images
                    print(image_paths)
                    if image_paths:
                        # # Number of columns per row
                        # num_columns_per_row = min(len(image_paths), 3)

                        # # Loop through the image paths and display them in rows of columns
                        # for i in range(0, len(image_paths), num_columns_per_row):
                        #     cols = st.columns(num_columns_per_row)
                        #     for j, image_path in enumerate(image_paths[i:i+num_columns_per_row]):
                        #         try:
                        #             with cols[j]:
                        #                 st.image(image_path, width=get_image_width(num_columns_per_row))
                        #         except Exception as e:
                        #             print(f"Failed to load {image_path}: {e}")
                        for image_path in image_paths:
                            try:
                                st.image(image_path, width=400)
                            except Exception as e:
                                print(f"Failed to load {image_path}: {e}")