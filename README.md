# 🎥 YouTube RAG Chatbot

An intelligent chatbot built with LangGraph and Streamlit that allows you to have conversations about YouTube videos using RAG (Retrieval-Augmented Generation).

## ✨ Features

- **🎬 YouTube Video Processing**: Extract and process transcripts from any YouTube video
- **💬 ChatGPT-Style Streaming**: Real-time token-by-token response streaming
- **🧠 RAG Architecture**: Semantic chunking with MMR retrieval for relevant context
- **📝 Thread Management**: Multiple conversation threads with auto-generated titles
- **🔄 Conversation Memory**: Persistent chat history with LangGraph checkpointing
- **🎯 Smart Context Retrieval**: Fetches only relevant transcript chunks for each question

## 🛠️ Tech Stack

- **Framework**: LangGraph for conversation orchestration
- **LLM**: OpenAI GPT-4o-mini with streaming
- **Vector DB**: ChromaDB for embeddings storage
- **Frontend**: Streamlit with custom UI
- **Embeddings**: OpenAI text-embedding-3-small
- **Database**: SQLite with WAL mode for thread management

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/akashcodejames/youtube_chat.git
cd youtube_chat
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp env_example.txt .env
# Edit .env and add your OpenAI API key
```

## 💻 Usage

1. **Start the application**
```bash
streamlit run youtube_rag_app.py
```

2. **Open your browser** at `http://localhost:8502`

3. **Process a YouTube video**
   - Paste any YouTube URL in the sidebar
   - Click "Process Video"
   - Wait for transcript processing (~30 seconds)

4. **Start chatting!**
   - Ask questions about the video content
   - Watch responses stream in real-time
   - Start new threads or switch between conversations

## 📁 Project Structure

```
youtube_chat/
├── youtube_rag_app.py          # Streamlit frontend
├── youtube_rag_backend.py      # LangGraph backend & RAG logic
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── youtube_rag.db             # SQLite database (auto-created)
└── chroma_db/                 # Vector database (auto-created)
```

## 🎯 How It Works

1. **Video Processing**
   - Extracts transcript using YouTube Transcript API
   - Splits into semantic chunks (350 tokens max)
   - Generates embeddings and stores in ChromaDB

2. **Question Answering**
   - Retrieves relevant chunks using MMR (Maximal Marginal Relevance)
   - Provides context to GPT-4o-mini
   - Streams response token-by-token

3. **Thread Management**
   - Each conversation has a unique thread ID
   - Auto-generates titles from first message
   - Persists history with LangGraph checkpointing

## 🔧 Configuration

Key settings in `youtube_rag_backend.py`:

```python
# LLM Configuration
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

# Chunking Parameters
similarity_threshold = 0.75  # Semantic similarity for grouping
max_tokens = 350            # Maximum chunk size

# Retrieval Parameters
k = 3                       # Number of chunks to retrieve
lambda_param = 0.6          # MMR diversity parameter
```

## 🐛 Troubleshooting

**Database Locked Error**
- Fixed with WAL mode and single connection
- Automatically handled in current version

**Transcript Not Found**
- Video may not have captions enabled
- Try a different video or check URL

**Streaming Not Working**
- Ensure `streaming=True` in ChatOpenAI config
- Check browser refresh if auto-reload fails

## 📝 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com)
- Transcripts from [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ using LangGraph and Streamlit
