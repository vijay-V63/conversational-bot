import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import json
import pandas as pd
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from faker import Faker  # For generating mock data
import plotly.express as px  # For visualizations
import requests
from PIL import Image
import io

# Load environment variables
load_dotenv()
fake = Faker()

# Hugging Face API settings
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def generate_image_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Hugging Face API error {response.status_code}: {response.text}")

    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    return image

# ========== SESSION STATE INITIALIZATION ==========
def initialize_session_state():
    defaults = {
        "chat_history": [],
        "conversation_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_domain": "General Knowledge",
        "show_history": False,
        "show_analytics": False,
        "saved_conversations": {},
        "user_profile": {
            "name": fake.name(),
            "expertise": fake.job(),
            "preferred_style": "Professional"
        },
        "ai_persona": "Helpful Expert",
        "active_tools": ["Web Search", "Code Interpreter"],
        "conversation_ratings": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ========== UI COMPONENTS ==========
def display_header():
    st.set_page_config(page_title="ZEUS AI", layout="wide", page_icon="🚀")
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("""
            <style>
                .typing {
                    border-right: 3px solid;
                    white-space: nowrap;
                    overflow: hidden;
                    animation: typing 2s steps(20, end), blink-caret .75s step-end infinite;
                }
                @keyframes typing {
                    from { width: 0 }
                    to { width: 100% }
                }
                @keyframes blink-caret {
                    from, to { border-color: transparent }
                    50% { border-color: orange; }
                }
            </style>
            <h1 class="typing">ZEUS AI Assistant</h1>
            """, unsafe_allow_html=True)
            st.caption("Your hyper-intelligent, multi-modal AI companion")
        with col2:
            st.image("logo.png", width=550)

def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("---")
        st.subheader("AI Configuration")
        model = st.selectbox(
            "Model",
            [
                "meta-llama/llama-4-scout:free",
                "meta-llama/llama-4-maverick:free",
                "deepseek/deepseek-chat-v3.1:free",
                "google/gemini-2.0-flash-exp:free",
                "mistralai/mistral-small-3.2-24b-instruct:free",
                "nvidia/nemotron-nano-9b-v2:free"
            ],
            index=0,
            help="Select the AI engine powering your assistant"
        )
        st.session_state.ai_persona = st.selectbox(
            "AI Persona",
            ["Helpful Expert", "Creative Genius", "Technical Specialist", "Friendly Advisor"],
            index=0,
        )
        st.subheader("👤 Your Profile")
        st.session_state.user_profile["preferred_style"] = st.selectbox(
            "Response Style",
            ["Professional", "Concise", "Detailed", "Casual"],
            index=0,
        )
        st.session_state.current_domain = st.selectbox(
            "🧠 Knowledge Focus",
            [
                "General Knowledge",
                "Technical/IT",
                "Business",
                "Scientific",
                "Creative Arts",
                "Legal",
                "Medical",
            ],
            index=0,
        )
        st.subheader("🛠️ Active Tools")
        tools = st.multiselect(
            "Select tools to enable:",
            options=[
                "Web Search",
                "Code Interpreter",
                "Data Analysis",
                "Document Reader",
                "Image Generator",
            ],
            default=st.session_state.active_tools,
        )
        st.session_state.active_tools = tools
        st.markdown("---")
        st.subheader("📂 Navigation")
        if st.button("📜 Conversation History"):
            st.session_state.show_history = not st.session_state.show_history
        if st.button("📊 Chat Analytics"):
            st.session_state.show_analytics = not st.session_state.show_analytics
        st.markdown("---")
        st.markdown(f"**Session Started:** {st.session_state.conversation_start_time}")
        st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
        return model

def display_full_history():
    st.title("📜 Full Conversation History")
    st.write(f"Conversation started at: {st.session_state.conversation_start_time}")
    if not st.session_state.chat_history:
        st.info("No conversation history yet.")
    else:
        search_term = st.text_input("🔍 Search conversations...")
        filtered_history = (
            [msg for msg in st.session_state.chat_history if search_term.lower() in msg["content"].lower()]
            if search_term else st.session_state.chat_history
        )
        for msg in filtered_history:
            with st.chat_message(name=msg["role"]):
                st.write(msg["content"])
        st.download_button(
            "💾 Export as JSON",
            data=json.dumps(st.session_state.chat_history, indent=2),
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d')}.json",
        )
    if st.button("⬅️ Back to Chat"):
        st.session_state.show_history = False
        st.rerun()

def display_analytics():
    st.title("📊 Conversation Analytics")
    if not st.session_state.chat_history:
        st.warning("No data to analyze yet")
        return
    df = pd.DataFrame(st.session_state.chat_history)
    df["length"] = df["content"].apply(len)
    df["time"] = pd.to_datetime(df.get("timestamp", datetime.now()))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(df))
    with col2:
        st.metric("Your Messages", sum(df["role"] == "human"))
    with col3:
        st.metric("AI Messages", sum(df["role"] == "AI"))
    tab1, tab2 = st.tabs(["Message Length", "Activity Over Time"])
    with tab1:
        fig = px.histogram(df, x="length", color="role", title="Distribution of Message Lengths")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        time_df = df.groupby([df["time"].dt.hour, "role"]).size().unstack()
        fig = px.line(time_df, title="Message Activity by Hour")
        st.plotly_chart(fig, use_container_width=True)
    if st.button("⬅️ Back to Chat"):
        st.session_state.show_analytics = False
        st.rerun()

def generate_response(user_input, zeus_chat, memory):
    system_prompt = f"""
    You are zeus ai - a highly advanced AI assistant. 
    Current Mode: {st.session_state.current_domain}
    User Profile: {st.session_state.user_profile}
    Active Tools: {st.session_state.active_tools}
    Respond as a {st.session_state.ai_persona} with a {st.session_state.user_profile['preferred_style']} style.
    """
    if "Technical" in st.session_state.current_domain:
        system_prompt += "\nProvide detailed, accurate technical information with examples when possible."
    elif "Creative" in st.session_state.current_domain:
        system_prompt += "\nBe imaginative and original in your responses."
    
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )
    
    conversation = LLMChain(
        llm=zeus_chat,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    with st.spinner("zeus ai is thinking..."):
        start_time = time.time()
        response = conversation.invoke({"human_input": user_input})
        processing_time = time.time() - start_time
    
    # Extract the response text from the result
    response_text = response["text"] if "text" in response else str(response)
    response_text += f"\n\n*[Generated in {processing_time:.2f}s | {st.session_state.current_domain} Mode]*"
    
    return response_text

def main_chat_interface(zeus_chat, memory):
    st.markdown(f"### 💬 Chat - **{st.session_state.current_domain}** Mode")
    st.caption(f"Persona: {st.session_state.ai_persona} | Style: {st.session_state.user_profile['preferred_style']}")
    
    # Display recent chat history
    for msg in st.session_state.chat_history[-4:]:
        with st.chat_message(name=msg["role"]):
            st.write(msg["content"])
    
    # Chat input form
    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100, placeholder="Type your message or upload a file...")
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("🚀 Send")
        with col2:
            enhance_clicked = st.form_submit_button("✨ Enhance")
    
    if enhance_clicked and user_input:
        user_input = f"[ENHANCED QUERY]: {user_input}\n\nPlease expand on this with additional details and examples."
        submitted = True
    
    if submitted and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "human", 
            "content": user_input, 
            "timestamp": datetime.now().isoformat()
        })
        
        # Handle image generation
        if "Image Generator" in st.session_state.active_tools and user_input.lower().startswith("generate image"):
            prompt = user_input.lower().replace("generate image", "").strip()
            try:
                image = generate_image_huggingface(prompt)
                st.session_state.chat_history.append({
                    "role": "AI",
                    "content": f"Generated image for prompt: {prompt}",
                    "timestamp": datetime.now().isoformat(),
                })
                with st.chat_message("AI"):
                    st.image(image, caption=prompt)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
        else:
            # Generate AI response
            response = generate_response(user_input, zeus_chat, memory)
            st.session_state.chat_history.append({
                "role": "AI", 
                "content": response, 
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()

def main():
    initialize_session_state()
    display_header()
    
    if st.session_state.show_history:
        display_full_history()
        return
        
    if st.session_state.show_analytics:
        display_analytics()
        return
        
    model = setup_sidebar()
    
    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            st.error("API key not found. Please configure your .env file.")
            return
            
        # Initialize the chat model
        zeus_chat = ChatOpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0.7 if "Creative" in st.session_state.current_domain else 0.3,
        )
        
        # Initialize memory
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Load existing chat history into memory
        for msg in st.session_state.chat_history:
            if msg["role"] == "human":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])
        
        main_chat_interface(zeus_chat, memory)
        
    except Exception as e:
        st.error(f"System error: {str(e)}")
        st.info("Please check your API keys and internet connection.")

if __name__ == "__main__":
    main()
