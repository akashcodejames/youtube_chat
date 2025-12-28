import streamlit as st
import youtube_rag_backend as backend
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="YouTube RAG Assistant", 
    page_icon="🎥", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        text-align: left;
        border-radius: 5px;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
    }
    .video-status {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION INITIALIZATION ====================

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if 'video_processed' not in st.session_state:
    st.session_state['video_processed'] = False

if 'processing_video' not in st.session_state:
    st.session_state['processing_video'] = False

# ==================== UTILITY FUNCTIONS ====================

def start_new_chat():
    """Start a new conversation thread"""
    st.session_state['thread_id'] = str(uuid.uuid4())
    st.session_state['video_processed'] = False
    st.session_state['processing_video'] = False

def get_messages_for_thread(thread_id):
    """Fetch message history from LangGraph state"""
    config = {'configurable': {'thread_id': thread_id}}
    try:
        state = backend.chatbot.get_state(config)
        return state.values.get('messages', [])
    except:
        return []

def check_if_video_processed(thread_id):
    """Check if thread has a processed video"""
    video_id = backend.get_thread_video_id(thread_id)
    return video_id is not None

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("🎥 YouTube RAG")
    
    # New Chat Button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        start_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # YouTube URL Input
    st.subheader("Process YouTube Video")
    youtube_url = st.text_input(
        "Enter YouTube URL", 
        placeholder="https://youtube.com/watch?v=...",
        key="youtube_url_input"
    )
    
    if st.button("🔄 Process Video", use_container_width=True):
        if youtube_url:
            st.session_state['processing_video'] = True
            with st.spinner("Processing video... This may take a minute."):
                success = backend.process_youtube_video(
                    youtube_url, 
                    st.session_state['thread_id']
                )
                
                if success:
                    st.session_state['video_processed'] = True
                    st.success("✅ Video processed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process video. Check URL and try again.")
            st.session_state['processing_video'] = False
        else:
            st.warning("Please enter a YouTube URL")
    
    st.markdown("---")
    
    # Thread History
    st.caption("📜 Recent Conversations")
    existing_threads = backend.get_all_thread_ids()
    
    for t_id in existing_threads:
        title = backend.get_thread_title(t_id)
        
        if t_id == st.session_state['thread_id']:
            st.markdown(f"**🔹 {title}**")
        else:
            if st.button(title, key=f"thread_{t_id}", use_container_width=True):
                st.session_state['thread_id'] = t_id
                st.session_state['video_processed'] = check_if_video_processed(t_id)
                st.rerun()

# ==================== MAIN CHAT INTERFACE ====================

st.title("🤖 YouTube RAG Assistant")

# Check video status for current thread
current_video_id = backend.get_thread_video_id(st.session_state['thread_id'])
if current_video_id:
    st.session_state['video_processed'] = True
    st.info(f"📹 Video loaded: `{current_video_id}` | Ask questions about the video!")
else:
    st.warning("⚠️ No video processed yet. Enter a YouTube URL in the sidebar to get started.")

# Load messages for current thread
current_messages = get_messages_for_thread(st.session_state['thread_id'])

# Display chat history
for msg in current_messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
if user_input := st.chat_input("Ask a question about the video..."):
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Check if first message
    is_first_message = len(current_messages) == 0
    
    # Generate assistant response
    with st.chat_message("assistant"):
        stream_container = st.empty()
        full_response = ""
        
        config = {
            'configurable': {
                'thread_id': st.session_state['thread_id']
            }
        }
        
        # Add thread_id to state
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "thread_id": st.session_state['thread_id']
        }
        
        # Stream response - SAME PATTERN AS WORKING CHATBOT
        try:
            for chunk, _ in backend.chatbot.stream(
                input_state,
                config=config,
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    full_response += chunk.content
                    stream_container.markdown(full_response + "▌")
            
            stream_container.markdown(full_response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Generate title for first message
    if is_first_message:
        with st.spinner("Generating title..."):
            new_title = backend.generate_chat_name(user_input)
            backend.save_thread_title(
                st.session_state['thread_id'], 
                new_title,
                youtube_url=None,
                video_id=current_video_id
            )
    else:
        backend.update_thread_timestamp(st.session_state['thread_id'])
    
    st.rerun()

# ==================== FOOTER ====================

st.markdown("---")
st.caption("💡 Tip: Process a YouTube video first, then ask questions about its content!")