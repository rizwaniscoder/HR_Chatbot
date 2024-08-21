import streamlit as st
from langchain import WikipediaAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
import toml

def load_api_keys(file_path):
    try:
        secrets = toml.load(file_path)
        api_key = secrets["OPENAI_API_KEY"]
        serper_api_key = secrets["SERPAPI_API_KEY"]
        return api_key, serper_api_key
    except KeyError as e:
        st.error(f"API key missing: {e}")
        raise
    except toml.TomlDecodeError as e:
        st.error(f"Error loading TOML file: {e}")
        raise

# Initialize tools and agent with prompt template
def initialize_tools_and_agent(api_key, serper_api_key):
    try:
        wikipedia = WikipediaAPIWrapper()
        serpapi = SerpAPIWrapper(serpapi_api_key=serper_api_key)

        tools = [
            Tool(name="Wikipedia", func=wikipedia.run, description="Search Wikipedia for information."),
            Tool(name="SerpApi", func=serpapi.run, description="Search the web using SerpApi.")
        ]

        llm = OpenAI(temperature=0.7, api_key=api_key)
        
        # Define prompt template
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert HR Compliance Agent. Your responses should only address questions related to HR compliance, such as labor laws, employment laws, and other HR-related regulations. If the question is not related to HR compliance, ask the user to provide a more specific question or related keywords.

            User question: {query}
            """
        )
        
        # Initialize the agent with the prompt template
        agent = initialize_agent(
            tools, 
            llm, 
            agent_type="zero-shot-react-description",
            prompt_template=prompt_template,
            verbose=True
        )
        
        return agent
    except Exception as e:
        st.error(f"Error initializing tools or agent: {e}")
        raise


def display_chat_history():
    try:
        for chat in st.session_state.chat_history:
            with st.chat_message("assistant" if chat["role"] == "assistant" else "user"):
                st.markdown(chat["content"])
    except Exception as e:
        st.error(f"Error displaying chat history: {e}")


def process_user_input(agent, user_question):
    try:
        # Construct the prompt using the template
        prompt = f"""
        You are an expert HR Compliance Agent. Your responses should only address questions related to HR compliance, such as labor laws, employment laws, and other HR-related regulations. If the question is not related to HR compliance, ask the user to provide a more specific question or related keywords.

        User question: {user_question}
        """
        
        # Run the agent with the constructed prompt and handle parsing errors
        response = agent.run(input=prompt, handle_parsing_errors=True)
        
        # Check if the response needs clarification
        if "provide a more specific question" in response:
            response = "Could you please provide more specific details or keywords related to HR compliance?"
        return response
    except Exception as e:
        st.error(f"Error processing user question: {e}")
        return "Sorry, there was an error processing your question."

def main():
    api_key, serper_api_key = load_api_keys("secrets.toml")

    # Initialize Streamlit configuration
    st.set_page_config(page_title="HR Compliance Chatbot", layout="wide")
    st.title("HR Compliance Advisor")
    st.subheader("Interact with the Bot for Reliable Compliance Answers")

    # Initialize session state for storing chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    with st.sidebar:
        st.markdown("""
            <style>
            .css-1d391kg {
                background-color: #f0f2f6;
                color: #333;
            }
            .css-1y0t0cm {
                background-color: #4a90e2;
                color: white;
            }
            .css-1nb8a2x {
                color: #4a90e2;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("### Information")
        st.markdown("Ask your HR compliance questions and get answers from the latest sources.")

        if st.button("New Chat"):
            st.session_state.chat_history = []

    # Initialize the agent
    agent = initialize_tools_and_agent(api_key, serper_api_key)

    # Display chat history
    display_chat_history()

    # Streamlit user input
    if user_question := st.chat_input("Ask a compliance question:"):
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        # Process the question
        response = process_user_input(agent, user_question)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
