# YouTube RAG Assistant with LangGraph

A full-featured RAG (Retrieval-Augmented Generation) application that processes YouTube videos, stores embeddings in a vector database, and allows users to ask questions about video content using advanced MMR (Maximal Marginal Relevance) retrieval.

## Features

- **YouTube Video Processing**: Automatically fetches transcripts and processes them
- **Semantic Hybrid Chunking**: Intelligently chunks text based on semantic similarity and token limits
- **MMR Retrieval**: Uses Maximal Marginal Relevance for diverse and relevant context retrieval
- **Vector Database**: ChromaDB for persistent embedding storage
- **Conversation Persistence**: SQLite checkpointer maintains chat history across sessions
- **Multi-Thread Support**: Manage multiple video conversations simultaneously
- **Streaming Responses**: Real-time response generation
- **Auto-Generated Titles**: Conversations automatically titled based on first message

## Architecture

### LangGraph Flow

```
START → retrieve_context → chat → END
```

1. **retrieve_context node**: Retrieves relevant chunks from ChromaDB using MMR
2. **chat node**: Generates response using retrieved context and conversation history

### Components

- **youtube_rag_backend.py**: Core logic including LangGraph, vector DB, and YouTube processing
- **app.py**: Streamlit UI with sidebar navigation and chat interface
- **ChromaDB**: Vector database (stored in `./chroma_db`)
- **SQLite**: Conversation checkpoints and metadata (stored in `youtube_rag.db`)

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd youtube-rag-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
```

4. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=sk-...
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Workflow

1. **Start New Chat**: Click "➕ New Chat" in the sidebar
2. **Process Video**: 
   - Paste a YouTube URL in the sidebar
   - Click "🔄 Process Video"
   - Wait for processing to complete (this downloads transcript, chunks it, generates embeddings, and stores in ChromaDB)
3. **Ask Questions**: Type questions about the video content in the chat input
4. **Resume Conversations**: Click on any previous conversation in the sidebar to resume

## Technical Details

### Semantic Hybrid Chunking

The application uses a sophisticated chunking strategy that:
- Splits text into sentences
- Computes embeddings for each sentence
- Groups sentences based on semantic similarity (cosine similarity threshold: 0.75)
- Respects token limits (max 350 tokens per chunk)
- Creates semantically coherent chunks

```python
def semantic_hybrid_chunking(
    text: str,
    similarity_threshold: float = 0.75,
    max_tokens: int = 350
) -> List[str]
```

### MMR Retrieval

MMR balances relevance and diversity:
- **λ parameter**: 0.6 (60% relevance, 40% diversity)
- Selects top k=3 chunks
- Prevents redundant information

```python
score = λ * similarity - (1 - λ) * diversity
```

### State Management

**ChatState Schema:**
```python
class ChatState(TypedDict):
    messages: list[BaseMessage]  # Conversation history
    context: str                 # Retrieved context from vector DB
    thread_id: str              # Unique thread identifier
```

### Database Schema

**SQLite (thread_titles table):**
- `thread_id`: Unique conversation ID
- `title`: Auto-generated conversation title
- `youtube_url`: Original YouTube URL
- `video_id`: Extracted video ID
- `updated_at`: Last activity timestamp

**ChromaDB Collections:**
- One collection per thread: `thread_{thread_id}`
- Stores document chunks with embeddings
- Metadata includes video_id and URL

## Configuration

### Adjustable Parameters

In `youtube_rag_backend.py`:

```python
# Chunking
similarity_threshold = 0.75  # Semantic similarity threshold
max_tokens = 350            # Max tokens per chunk

# MMR Retrieval
k = 3                       # Number of chunks to retrieve
lambda_param = 0.6         # Relevance vs diversity balance

# LLM
model = "gpt-4o-mini"      # OpenAI model
temperature = 0.7          # Response randomness
```

## File Structure

```
youtube-rag-assistant/
├── youtube_rag_backend.py    # Core backend logic
├── app.py                    # Streamlit UI
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── README.md                # This file
├── youtube_rag.db           # SQLite database (created on run)
└── chroma_db/               # ChromaDB storage (created on run)
```

## Advanced Features

### Custom Query Generation

The system automatically enhances user queries for better retrieval by using the full conversation context.

### Conversation Persistence

All conversations are saved and can be resumed at any time. The SQLite checkpointer stores:
- Complete message history
- Retrieved contexts
- Thread metadata

### Thread Management

- Threads are sorted by most recent activity
- Each thread is associated with one YouTube video
- Can create unlimited threads for different videos

## Troubleshooting

### Video Processing Fails

- Ensure the YouTube video has captions/transcripts
- Check your internet connection
- Verify the URL format is correct

### API Errors

- Verify your OpenAI API key is correct
- Check you have sufficient API credits
- Ensure `.env` file is in the root directory

### Database Issues

- Delete `youtube_rag.db` and `chroma_db/` folder to reset
- Ensure write permissions in the directory

## Performance Tips

1. **Chunk Size**: Decrease `max_tokens` for faster processing, increase for more context
2. **Retrieval Count**: Adjust `k` parameter (fewer chunks = faster, more chunks = more context)
3. **MMR Lambda**: Increase for more relevance, decrease for more diversity

## Future Enhancements

- [ ] Support for multiple videos per conversation
- [ ] PDF and document upload support
- [ ] Custom embedding models
- [ ] Query history and suggestions
- [ ] Export conversations
- [ ] Video timestamp references in responses

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Credits

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://openai.com/)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)