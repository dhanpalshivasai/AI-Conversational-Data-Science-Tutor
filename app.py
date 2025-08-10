import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

st.title("🧠 AI Conversational Data Science Tutor")

# Initialize memory for conversation history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

memory = st.session_state.memory

# Initialize AI model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.secrets["api_key"])

# Sidebar - Conversation History
st.sidebar.subheader("🕒 Conversation History")
for msg in memory.chat_memory.messages:
    st.sidebar.write(f"**{msg.type.capitalize()}:** {msg.content}")

# User input for text-based questions using chat_input()
user_input = st.chat_input("❓ Ask your Data Science question:")

# Define AI prompt template
prompt = PromptTemplate(
    input_variables=["user_input", "history"],
    template="""
    You are an AI-powered Data Science tutor.
    Designed to assist users with their Data Science-related queries.
    Your goal is to provide **detailed, structured, and well-explained answers** to user questions while maintaining the context of previous conversations.
    Provide Well-Structured Answers, including **definitions, explanations, and real-world examples.** If relevant, suggest best practices, algorithms, or tools used in Data Science.  
    Encourage Hands-on Learning by providing code snippets or pseudocode if applicable, and suggest relevant libraries (e.g., NumPy, Pandas, Scikit-Learn, TensorFlow).  

    ***Respond ONLY to Data Science-related queries. If the user asks something unrelated, politely guide them back to Data Science topics.***


    **Conversation History:**  
    {history}  
    
    **User Question:**  
    {user_input}  
    
    **AI Response:**  
    """
)

# AI processing chain
chain = prompt | llm | RunnablePassthrough()

# Process user input
if user_input:
    try:
        history = "\n".join([
            f"User: {msg.content}" if msg.type == "human" else f"Tutor: {msg.content}"
            for msg in memory.chat_memory.messages
        ])

        # Generate response from AI
        response = chain.invoke({
            "user_input": user_input,
            "history": history
        })

        response_text = response if isinstance(response, str) else response.get("text", "Sorry, I couldn't process that.")

        # Display response with formatting
        st.markdown("**🧑‍🏫 Tutor:**")
        st.markdown(f"<div style='background-color:#f4f4f4; padding:10px; border-radius:5px;'>{response_text}</div>", unsafe_allow_html=True)

        # Add to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response_text)

    except Exception as e:
        st.error(f"❌ Error: {e}")

