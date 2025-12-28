from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
from youtube_transcript_api import YouTubeTranscriptApi
import re
import tiktoken

load_dotenv()

# ==================== CONFIGURATION ====================
DB_PATH = 'youtube_rag.db'
CHROMA_PATH = './chroma_db'
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

# Initialize shared SQLite connection with proper settings
db_conn = sqlite3.connect(database=DB_PATH, check_same_thread=False, timeout=30.0)
db_conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode for better concurrency
db_conn.row_factory = sqlite3.Row

# ==================== UTILITY FUNCTIONS ====================

def extract_youtube_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def approximate_token_count(text: str) -> int:
    """Estimate token count"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        return len(text.split()) * 1.3

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_hybrid_chunking(
    text: str,
    similarity_threshold: float = 0.75,
    max_tokens: int = 350
) -> List[str]:
    """Chunk text semantically with token limits"""
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    sentence_embeddings = embeddings_model.embed_documents(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = np.array(sentence_embeddings[0])
    
    for i in range(1, len(sentences)):
        next_sentence = sentences[i]
        next_embedding = np.array(sentence_embeddings[i])
        
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1),
            next_embedding.reshape(1, -1)
        )[0][0]
        
        tentative_chunk = " ".join(current_chunk + [next_sentence])
        
        if (
            similarity >= similarity_threshold
            and approximate_token_count(tentative_chunk) <= max_tokens
        ):
            current_chunk.append(next_sentence)
            current_embedding = np.mean([current_embedding, next_embedding], axis=0)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_embedding = next_embedding
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def mmr_retrieval(
    query_embedding,
    doc_embeddings,
    docs,
    k=3,
    lambda_param=0.6
):
    """MMR retrieval for diversity"""
    selected = []
    selected_indices = []
    
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    for _ in range(min(k, len(docs))):
        if not selected:
            idx = np.argmax(similarities)
        else:
            mmr_scores = []
            for i in range(len(docs)):
                if i in selected_indices:
                    mmr_scores.append(-1)
                    continue
                
                diversity = max(
                    cosine_similarity(
                        [doc_embeddings[i]],
                        [doc_embeddings[j] for j in selected_indices]
                    )[0]
                )
                
                score = (
                    lambda_param * similarities[i]
                    - (1 - lambda_param) * diversity
                )
                mmr_scores.append(score)
            
            idx = np.argmax(mmr_scores)
        
        selected.append(docs[idx])
        selected_indices.append(idx)
    
    return selected

# ==================== DATABASE FUNCTIONS ====================

def init_databases():
    """Initialize SQLite for checkpoints and titles"""
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            youtube_url TEXT,
            video_id TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db_conn.commit()

def save_thread_title(thread_id, title, youtube_url=None, video_id=None):
    db_conn.execute("""
        INSERT OR REPLACE INTO thread_titles (thread_id, title, youtube_url, video_id, updated_at) 
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (str(thread_id), title, youtube_url, video_id))
    db_conn.commit()

def update_thread_timestamp(thread_id):
    db_conn.execute("""
        UPDATE thread_titles 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE thread_id = ?
    """, (str(thread_id),))
    db_conn.commit()

def get_thread_title(thread_id):
    cursor = db_conn.execute(
        "SELECT title FROM thread_titles WHERE thread_id = ?", 
        (str(thread_id),)
    )
    result = cursor.fetchone()
    return result[0] if result else "New Chat"

def get_thread_video_id(thread_id):
    cursor = db_conn.execute(
        "SELECT video_id FROM thread_titles WHERE thread_id = ?",
        (str(thread_id),)
    )
    result = cursor.fetchone()
    return result[0] if result else None

def get_all_thread_ids():
    cursor = db_conn.execute("""
        SELECT thread_id FROM thread_titles 
        ORDER BY updated_at DESC
    """)
    return [row[0] for row in cursor.fetchall()]

# ==================== YOUTUBE & VECTOR DB ====================

def process_youtube_video(url: str, thread_id: str) -> bool:
    """Download transcript, chunk, embed, and store in ChromaDB"""
    try:
        video_id = extract_youtube_id(url)
        if not video_id:
            print(f"ERROR: Could not extract video ID from URL: {url}")
            return False
        
        print(f"Processing video ID: {video_id}")
        
        # Create API instance
        ytt_api = YouTubeTranscriptApi()
        
        # Try to fetch transcript with multiple language options
        transcript_data = None
        fetched_transcript = None
        
        # Method 1: Try fetching with common languages
        for lang_codes in [['en'], ['hi'], ['es'], ['fr'], ['de', 'en'], ['en', 'hi']]:
            try:
                print(f"Trying to fetch transcript with languages: {lang_codes}")
                fetched_transcript = ytt_api.fetch(video_id, languages=lang_codes)
                if fetched_transcript:
                    transcript_data = fetched_transcript.to_raw_data()
                    print(f"Successfully fetched transcript in {fetched_transcript.language} with {len(transcript_data)} entries")
                    break
            except Exception as lang_error:
                continue
        
        # Method 2: If still no transcript, try listing and selecting
        if not transcript_data:
            try:
                print("Trying list() approach...")
                transcript_list = ytt_api.list(video_id)
                
                # Try to find English transcript first
                try:
                    transcript_obj = transcript_list.find_transcript(['en', 'hi', 'es', 'de', 'fr'])
                    fetched_transcript = transcript_obj.fetch()
                    transcript_data = fetched_transcript.to_raw_data()
                    print(f"Fetched {len(transcript_data)} entries using list() method")
                except Exception as find_error:
                    print(f"find_transcript failed: {find_error}")
            except Exception as list_error:
                print(f"list() approach failed: {type(list_error).__name__}: {list_error}")
        
        # If still no transcript, give up
        if not transcript_data:
            print(f"ERROR: Could not fetch transcript for video {video_id} using any method")
            return False
        
        # Convert transcript data to text
        full_text = " ".join([entry['text'] for entry in transcript_data])
        
        if not full_text.strip():
            print("ERROR: Transcript is empty")
            return False
        
        print(f"Transcript length: {len(full_text)} characters")
        
        # Chunk the transcript
        chunks = semantic_hybrid_chunking(full_text)
        
        if not chunks:
            print("ERROR: No chunks generated from transcript")
            return False
        
        print(f"Generated {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = embeddings_model.embed_documents(chunks)
        print(f"Generated embeddings for {len(chunks)} chunks")
        
        # Store in ChromaDB
        collection_name = f"thread_{thread_id}"
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"video_id": video_id, "url": url}
        )
        
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        
        print(f"Successfully stored in ChromaDB collection: {collection_name}")
        
        # Save metadata
        save_thread_title(thread_id, f"YouTube: {video_id[:8]}...", url, video_id)
        
        return True
    
    except Exception as e:
        print(f"ERROR processing video: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def retrieve_from_vectordb(thread_id: str, query: str, k: int = 3) -> str:
    """Retrieve context using MMR from ChromaDB"""
    try:
        collection_name = f"thread_{thread_id}"
        collection = chroma_client.get_collection(collection_name)
        
        # Get all documents and embeddings
        all_docs = collection.get(include=['documents', 'embeddings'])
        
        if not all_docs['documents']:
            return ""
        
        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)
        
        # Apply MMR
        doc_embeddings = np.array(all_docs['embeddings'])
        selected_docs = mmr_retrieval(
            query_embedding,
            doc_embeddings,
            all_docs['documents'],
            k=k,
            lambda_param=0.6
        )
        
        return "\n\n".join(selected_docs)
    
    except Exception as e:
        print(f"Error retrieving from vector DB: {e}")
        return ""

# ==================== LANGGRAPH STATE & NODES ====================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    thread_id: str

def retrieve_context_node(state: ChatState):
    """Node to retrieve context from vector DB"""
    messages = state["messages"]
    thread_id = state.get("thread_id", "")
    
    # Get last user message
    user_query = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        None
    )
    
    if not user_query or not thread_id:
        return {"context": ""}
    
    # Check if video exists for this thread
    video_id = get_thread_video_id(thread_id)
    if not video_id:
        return {"context": ""}
    
    # Retrieve context using MMR
    context = retrieve_from_vectordb(thread_id, user_query, k=3)
    return {"context": context}

def chat_node(state: ChatState):
    """Node to generate response with context"""
    messages = state['messages']
    context = state.get('context', '')
    
    # Build system message with context
    if context:
        system_msg = SystemMessage(content=f"""You are a helpful assistant answering questions about a YouTube video.
        
Use the following context from the video transcript to answer the user's question:

{context}

If the context doesn't contain relevant information, say so politely.""")
        messages_with_context = [system_msg] + messages
    else:
        messages_with_context = messages
    
    response = llm.invoke(messages_with_context)
    return {"messages": [response]}

# ==================== GRAPH SETUP ====================

# Use the same shared connection for checkpointer to avoid locking
checkpointer = SqliteSaver(conn=db_conn)

graph = StateGraph(ChatState)
graph.add_node("retrieve_context", retrieve_context_node)
graph.add_node("chat", chat_node)

graph.add_edge(START, "retrieve_context")
graph.add_edge("retrieve_context", "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)

# Initialize databases
init_databases()

# ==================== HELPER FOR STREAMLIT ====================

def generate_chat_name(user_message_content: str) -> str:
    """Generate chat title from first message"""
    system_prompt = "Summarize this message into a 4-5 word title. No quotes."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message_content)
    ]
    response = llm.invoke(messages)
    return response.content.strip()