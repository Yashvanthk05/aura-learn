# AuraLearn - Complete Project Workflow Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Backend API Endpoints](#backend-api-endpoints)
6. [Core Services](#core-services)
7. [Frontend Structure](#frontend-structure)
8. [Data Storage & Management](#data-storage--management)
9. [Complete User Workflows](#complete-user-workflows)

---

## Project Overview

**AuraLearn** is a professional document intelligence API that enables end-to-end PDF/document processing and conversion to audiobooks. It provides AI-powered summarization, intelligent search, and natural speech synthesis capabilities.

### Key Features

- **PDF/Document Ingestion**: Support for PDF, PPTX, DOCX, TXT, MD, CSV files
- **Extractive Summarization**: BiLSTM-based neural model for key sentence extraction
- **Abstractive Summarization**: Fine-tuned T5 transformer for semantic summary generation
- **Audiobook Generation**: High-quality TTS (Text-to-Speech) conversion
- **Hybrid Search**: Multi-algorithm search using FAISS, BM25, and TF-IDF
- **Chat Interface**: RAG (Retrieval Augmented Generation) with source citations
- **Explainable AI (XAI)**: Visualization of model decisions and reasoning
- **Audio Transcription**: Convert audio/video to text with automatic summarization

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React/Vite)                    │
│  - OAuth Google Login  │ Document Upload │ Chat Interface       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    HTTP/REST API (FastAPI)
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              BACKEND - Application Server                        │
│  [Lifespan Manager] ─ Initialize Services on Startup            │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  API Controllers │ │   Core Services  │ │   ML Services    │
│                  │ │                  │ │                  │
│ • Upload         │ │ • Security/Auth  │ │ • Extractive     │
│ • Document       │ │ • Configuration  │ │   Summarization  │
│ • Summarize      │ │                  │ │ • Abstractive    │
│ • Search         │ │                  │ │   Summarization  │
│ • Chat           │ │                  │ │ • Vector Store   │
│ • Audiobook      │ │                  │ │ • TTS/Audiobook  │
│ • Transcription  │ │                  │ │ • Transcription  │
│ • XAI Explain    │ │                  │ │ • XAI Services   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  File Storage    │ │  Vector Indexes  │ │  ML Models       │
│                  │ │                  │ │                  │
│ • uploads/       │ │ • FAISS Index    │ │ • Extractive     │
│ • outputs/       │ │ • BM25 Index     │ │   Model (.pt)    │
│ • data/chunks    │ │ • TF-IDF Matrix  │ │ • T5 Summarizer  │
│ • data/sessions  │ │                  │ │ • XTTS V2 (TTS)  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## Technology Stack

### Backend

| Component      | Technology         | Purpose                                     |
| -------------- | ------------------ | ------------------------------------------- |
| **Framework**  | FastAPI            | REST API server with automatic OpenAPI docs |
| **Server**     | Uvicorn            | ASGI application server                     |
| **Auth**       | JWT + Google OAuth | User authentication and authorization       |
| **Validation** | Pydantic           | Request/response schema validation          |

### Machine Learning

| Component                | Technology                               | Purpose                                   |
| ------------------------ | ---------------------------------------- | ----------------------------------------- |
| **Extractive Model**     | BiLSTM + Multi-Head Attention            | Select important sentences                |
| **Abstractive Model**    | Fine-tuned T5 Transformer                | Generate concise summaries                |
| **Embeddings**           | Sentence-Transformers (all-MiniLM-L6-v2) | Dense vector representations              |
| **Search - Dense**       | FAISS (Facebook AI Similarity Search)    | Fast semantic similarity matching         |
| **Search - Lexical**     | BM25                                     | Keyword-based document ranking            |
| **Search - Statistical** | TF-IDF                                   | Term frequency-inverse document frequency |
| **TTS**                  | Coqui TTS (XTTS V2)                      | Multilingual text-to-speech synthesis     |
| **Speech Recognition**   | OpenAI Whisper                           | Audio transcription                       |
| **Image Captioning**     | BLIP (Salesforce)                        | Caption PDF/document images               |

### Frontend

| Component       | Technology       | Purpose                       |
| --------------- | ---------------- | ----------------------------- |
| **Framework**   | React 19         | UI component library          |
| **Build Tool**  | Vite             | Fast build tooling            |
| **Styling**     | Tailwind CSS     | Utility-first CSS framework   |
| **OAuth**       | Google OAuth 2.0 | Single sign-on authentication |
| **HTTP Client** | Fetch API        | API communication             |

### Storage & Processing

| Component            | Technology     | Purpose                           |
| -------------------- | -------------- | --------------------------------- |
| **Document Parsing** | PyMuPDF (fitz) | Extract text and images from PDFs |
| **Presentations**    | python-pptx    | Parse PowerPoint files            |
| **Word Documents**   | python-docx    | Parse .docx files                 |
| **Serialization**    | JSON           | Store chunks, sessions, vectors   |
| **Linear Algebra**   | NumPy          | Vector operations                 |

---

## Data Flow Pipeline

### Complete End-to-End Flow

```
User Action → API Request → Validation → Processing → Response → Storage
     ↓             ↓              ↓           ↓            ↓          ↓
  UI Click    FastAPI Route   Pydantic   Service Layer  JSON/File  Disk/Memory
```

### Detailed Pipeline Stages

#### 1. **Document Upload & Processing Pipeline**

```
PDF File Uploaded
    ↓
[Upload Controller] - Saves file, validates extension
    ↓
[Document Service] - Calls pdf_processor
    ↓
[PDF Processor]
    ├─→ Extract text with metadata (page, font size)
    ├─→ Detect document structure (headers, paragraphs)
    ├─→ Extract and caption images
    ├─→ Perform text preprocessing (cleaning, tokenization)
    └─→ Create semantic chunks with topics
    ↓
JSON Chunks File Created
    ↓
[Vector Store Creation]
    ├─→ Generate embeddings using SentenceTransformer
    ├─→ Build FAISS index for semantic search
    ├─→ Build BM25 index for keyword matching
    └─→ Build TF-IDF matrix for statistical search
    ↓
Stored in: data/{document_id}_chunks.json
         + Vector indices in memory/cache
```

**Key Files Involved:**

- [main.py](main.py) - Application entry point
- [app/api/controllers/upload_controller.py](app/api/controllers/upload_controller.py#L17-L45) - Handles file upload
- [app/services/document_service.py](app/services/document_service.py#L49-L65) - Document processing logic
- [app/utils/pdf_processor.py](app/utils/pdf_processor.py#L55-L80) - PDF extraction and chunking

---

#### 2. **Search Pipeline (Hybrid Multi-Algorithm)**

```
User Query
    ↓
[Search Controller] - Extracts query and parameters
    ↓
[Vector Store Service]
    ├─→ Query Embedding Generation
    │   └─→ SentenceTransformer encodes query → 384-dim vector
    │
    ├─→ Dense Search (FAISS)
    │   ├─→ Compute L2 distance to all chunk embeddings
    │   ├─→ Return top K candidates with distance scores
    │   └─→ Normalize scores: 1 - (distance / max_distance)
    │
    ├─→ Lexical Search (BM25)
    │   ├─→ Tokenize query and corpus
    │   ├─→ Calculate BM25 relevance scores
    │   └─→ Normalize scores to [0, 1]
    │
    ├─→ Statistical Search (TF-IDF)
    │   ├─→ Compute TF-IDF representation of query
    │   ├─→ Calculate cosine similarity with all chunks
    │   └─→ Normalize scores to [0, 1]
    │
    └─→ Hybrid Fusion (Weighted Combination)
        ├─→ FAISS Weight: 0.5 (semantic meaning)
        ├─→ BM25 Weight: 0.3 (keyword presence)
        ├─→ TF-IDF Weight: 0.2 (term statistics)
        └─→ Combined Score = Σ(weight × normalized_score)
    ↓
Return Top-K results ranked by combined score
```

**Key Files:**

- [app/api/controllers/search_controller.py](app/api/controllers/search_controller.py#L10-L45) - Search endpoint
- [app/services/vector_store_service.py](app/services/vector_store_service.py#L12-L153) - Hybrid search implementation

---

#### 3. **Extractive Summarization Pipeline**

```
Input Text
    ↓
[Extractive Service]
    ├─→ NLTK Sentence Tokenization
    │   └─→ Split text into individual sentences
    │
    ├─→ Sentence Embedding
    │   └─→ SentenceTransformer encodes each sentence → 384-dim vectors
    │
    ├─→ Model Forward Pass (BiLSTM + Attention)
    │   ├─→ Positional Encoding (100 max positions)
    │   ├─→ Bidirectional LSTM (256 hidden, 2 layers)
    │   │   └─→ Captures long-range dependencies
    │   ├─→ Multi-Head Attention (8 heads)
    │   │   └─→ Learns which sentences are important
    │   ├─→ Feed-Forward Classifier
    │   │   └─→ Outputs importance score [0, 1] per sentence
    │   └─→ LayerNorm for stability
    │
    ├─→ Score Filtering
    │   ├─→ Get top-N sentences by importance score
    │   └─→ Sort by original position (maintain order)
    │
    └─→ Summary Assembly
        └─→ Join selected sentences with spaces
    ↓
Return Extractive Summary
```

**Key Architecture:**

- [app/services/extractive_service.py](app/services/extractive_service.py#L11-L60) - ExtractiveModel definition
- [app/services/extractive_service.py](app/services/extractive_service.py#L62-L120) - ExtractiveSummarizer class
- [app/api/controllers/summarize_controller.py](app/api/controllers/summarize_controller.py#L18-L50) - Summarize endpoint

---

#### 4. **Abstractive Summarization Pipeline**

```
Input Text
    ↓
[Abstractive Service]
    ├─→ Text Tokenization
    │   └─→ T5Tokenizer: "summarize: " + text
    │       (Prepend task prefix for T5)
    │
    ├─→ Input Preparation
    │   ├─→ Truncate to 512 tokens
    │   ├─→ Apply padding
    │   └─→ Generate attention mask
    │
    ├─→ T5 Model Decoding (Beam Search)
    │   ├─→ Beam Width: 8 beams (explore 8 most likely paths)
    │   ├─→ Beam Groups: 4 (diversity penalties)
    │   ├─→ Max Length: 150 tokens
    │   ├─→ Min Length: 40 tokens
    │   ├─→ No Repeat N-gram: 3 (prevent repetition)
    │   ├─→ Repetition Penalty: 3.0 (penalize repeated content)
    │   ├─→ Length Penalty: 2.0 (control length preference)
    │   ├─→ Diversity Penalty: 1.5 (encourage diverse beams)
    │   └─→ Top-P (Nucleus): 0.95 (probabilistic sampling)
    │
    ├─→ Token Decoding
    │   └─→ T5Tokenizer converts tokens back to text
    │       (skip special tokens like [CLS], [SEP])
    │
    └─→ Return Abstractive Summary
    ↓
Summary Text (novel, generated phrasing)
```

**Note:** Abstractive model generates new text, not from original corpus.

**Key Files:**

- [app/services/abstractive_service.py](app/services/abstractive_service.py#L1-L55) - AbstractiveSummarizer implementation
- [app/api/controllers/summarize_controller.py](app/api/controllers/summarize_controller.py#L18-L50) - Summarize endpoint

---

#### 5. **Text-to-Speech (Audiobook) Pipeline**

```
Summary Text
    ↓
[Audiobook Generation Service]
    ├─→ Coqui TTS Model (XTTS V2 - multilingual)
    │
    ├─→ Speaker Reference (Optional)
    │   ├─→ If provided: Use uploaded speaker WAV file
    │   └─→ If not provided: Use default reference audio
    │       └─→ Enables voice cloning (same voice style)
    │
    ├─→ Language Selection
    │   └─→ Set language code (e.g., 'en' for English)
    │
    ├─→ TTS Synthesis
    │   ├─→ Text analysis and phoneme generation
    │   ├─→ Mel-spectrogram prediction
    │   ├─→ Vocoder synthesis (convert to waveform)
    │   └─→ Output: WAV audio file
    │
    └─→ Save to: output/{audio_id}.wav
    ↓
Audio File URL returned to client
```

**Key Files:**

- [app/services/tts_service.py](app/services/tts_service.py#L1-L60) - AudiobookGenerator implementation
- [app/api/controllers/audiobook_controller.py](app/api/controllers/audiobook_controller.py#L10-L35) - Audiobook generation endpoint

---

#### 6. **Chat Interface with RAG Pipeline**

```
User Message (Query)
    ↓
[Chat Controller]
    ├─→ Retrieve relevant document chunks (Vector Store Search)
    │   └─→ Hybrid search returns top-3 most relevant chunks
    │
    ├─→ Build Context
    │   └─→ Format: "[Context 1] Topic: text ... [Context 2] ..."
    │
    ├─→ RAG Service generates response
    │   ├─→ If Abstractive: Use T5 to synthesize answer
    │   ├─→ If Extractive: Use BiLSTM to extract key information
    │   └─→ Fallback: Build responses from context
    │
    ├─→ Build Citations
    │   └─→ Create metadata for each source chunk
    │       ├─→ Citation ID, page number, topic
    │       ├─→ Relevance score and snippet
    │       └─→ Source document reference
    │
    ├─→ Add Citation Markers
    │   └─→ Insert [1], [2], [3] markers in response text
    │       (pointing to citations)
    │
    ├─→ Save to Session History
    │   ├─→ Store user message
    │   ├─→ Store assistant response with citations
    │   └─→ Save timestamp and metadata
    │
    └─→ Return ChatQueryResponse
        ├─→ Response text with citations
        ├─→ Citations array with full metadata
        └─→ Session ID and timestamp
    ↓
Display in Chat UI with clickable references
```

**Key Files:**

- [app/api/controllers/chat_controller.py](app/api/controllers/chat_controller.py#L1-L50) - Chat endpoints
- [app/services/chat_service.py](app/services/chat_service.py#L1-L100) - RAGChatService implementation
- [app/services/session_service.py](app/services/session_service.py#L1-L100) - Session persistence

---

#### 7. **Audio Transcription Pipeline**

```
Audio/Video File Uploaded
    ↓
[Transcription Controller]
    ├─→ Validate file type (MP3, WAV, MP4, etc.)
    │
    ├─→ Save temp file
    │
    ├─→ [Transcription Service]
    │   ├─→ Whisper Model (specified size: 'base')
    │   │   ├─→ Device: CPU or GPU
    │   │   ├─→ Language detection (optional)
    │   │   ├─→ Audio segmentation (find speech boundaries)
    │   │   └─→ Generate transcript with timestamps
    │   │
    │   └─→ Return: text, segments, language
    │
    ├─→ Automatic Summarization
    │   ├─→ Option 1: Extractive
    │   │   └─→ BiLSTM extracts key sentences
    │   ├─→ Option 2: Abstractive
    │   │   └─→ T5 generates summary
    │   └─→ Configurable num sentences/length
    │
    ├─→ Clean up temp file
    │
    └─→ Return TranscriptionResponse
        ├─→ Full transcript text
        ├─→ Automatic summary
        ├─→ Segment count and metadata
        └─→ Detected language
    ↓
Display transcript and summary to user
```

**Key Files:**

- [app/api/controllers/transcription_controller.py](app/api/controllers/transcription_controller.py#L12-L75) - Transcription endpoint
- [app/services/transcription_service.py](app/services/transcription_service.py) - Whisper transcription service

---

## Backend API Endpoints

### 1. **Authentication & Health**

#### Health Check

```http
GET /api/v1/health
```

- **Purpose**: Verify API status and model availability
- **Auth**: Required
- **Response**: [`HealthResponse`](app/models/schemas.py)
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "models_loaded": {
      "extractive": true,
      "abstractive": true,
      "tts": true
    }
  }
  ```

**Code Reference**: [health_controller.py](app/api/controllers/health_controller.py)

---

### 2. **Document Management**

#### Upload Document

```http
POST /api/v1/upload
Content-Type: multipart/form-data
```

- **Parameters**: `file` (PDF, PPTX, DOCX, TXT, MD, CSV)
- **Auth**: Required (JWT token)
- **Response**: [`UploadPDFResponse`](app/models/schemas.py)
  ```json
  {
    "document_id": "uuid-123...",
    "filename": "research_paper.pdf",
    "num_chunks": 45,
    "message": "Document processed successfully. 45 chunks extracted."
  }
  ```

**Workflow**:

1. File validation (extension, size)
2. Save to `uploads/` directory
3. Extract text and create chunks
4. Build vector indexes
5. Store document ownership

**Code Reference**: [upload_controller.py](app/api/controllers/upload_controller.py#L17-L45)

---

#### Get Document Info

```http
GET /api/v1/document/{document_id}
```

- **Response**: [`DocumentInfoResponse`](app/models/schemas.py)
  ```json
  {
    "document_id": "uuid-123...",
    "filename": "research_paper.pdf",
    "num_chunks": 45,
    "chunks": [
      {
        "chunk_id": 0,
        "topic": "Introduction",
        "page": 2,
        "text_preview": "This research paper investigates..."
      }
    ]
  }
  ```

**Code Reference**: [document_controller.py](app/api/controllers/document_controller.py#L10-L32)

---

#### List User Documents

```http
GET /api/v1/documents
```

- **Response**: Array of documents with metadata
- **Code Reference**: [document_controller.py](app/api/controllers/document_controller.py#L45-L48)

---

#### Delete Document

```http
DELETE /api/v1/document/{document_id}
```

- **Purpose**: Remove document and associated files
- **Lifecycle**: Deletes chunks, vectors, and uploaded file

**Code Reference**: [document_controller.py](app/api/controllers/document_controller.py#L35-L43)

---

### 3. **Search & Retrieval**

#### Hybrid Search

```http
POST /api/v1/search
Content-Type: application/json
```

- **Request** ([`SearchRequest`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "query": "What are the main findings?",
    "top_k": 5,
    "search_method": "hybrid" // hybrid | faiss | bm25 | tfidf
  }
  ```
- **Response** ([`SearchResponse`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "query": "What are the main findings?",
    "results": [
      {
        "chunk_id": 12,
        "topic": "Findings",
        "page": 15,
        "text": "Our analysis revealed...",
        "score": 0.892,
        "score_breakdown": {
          "faiss": 0.45,
          "bm25": 0.27,
          "tfidf": 0.17,
          "combined": 0.892
        }
      }
    ]
  }
  ```

**Search Methods**:

- **Hybrid** (default): Combines FAISS (50%), BM25 (30%), TF-IDF (20%)
- **FAISS**: Dense semantic embeddings only
- **BM25**: Keyword-based ranking
- **TF-IDF**: Statistical term frequency based

**Code Reference**: [search_controller.py](app/api/controllers/search_controller.py#L11-L48)

---

### 4. **Summarization**

#### Summarize Document

```http
POST /api/v1/summarize
Content-Type: application/json
```

- **Request** ([`SummarizeRequest`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "chunk_ids": [0, 1, 2], // optional: specific chunks
    "summarization_type": "extractive", // extractive | abstractive
    "num_sentences": 5, // for extractive
    "max_length": 150, // for abstractive
    "min_length": 40 // for abstractive
  }
  ```
- **Response** ([`SummarizeResponse`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "summarization_type": "extractive",
    "summary": "First important sentence. Second important sentence.",
    "num_chunks_processed": 3,
    "metadata": {
      "chunk_ids": [0, 1, 2]
    }
  }
  ```

**Types**:

- **Extractive**: Selects existing sentences (faster, preserves original language)
- **Abstractive**: Generates new text (slower, more concise)

**Code Reference**: [summarize_controller.py](app/api/controllers/summarize_controller.py#L18-L50)

---

#### Summarize & Generate Audio

```http
POST /api/v1/summarize-and-audio
Content-Type: multipart/form-data
```

- **Fields**:
  ```
  document_id: "uuid-123..."
  summarization_type: "abstractive"
  language: "en"
  chunk_ids: (optional) "[0, 1, 2]"
  num_sentences: 5
  max_length: 150
  min_length: 40
  speaker_audio: (optional) WAV file for voice cloning
  ```
- **Response** ([`SummarizeAndAudioResponse`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "summarization_type": "abstractive",
    "summary": "Executive summary...",
    "audio_url": "/api/v1/audio/audio-id.wav",
    "audio_filename": "audio-id.wav",
    "num_chunks_processed": 5
  }
  ```

**Workflow**:

1. Retrieve and combine specified chunks
2. Summarize text (extractive or abstractive)
3. Generate audio from summary (TTS)
4. Save to `outputs/` directory
5. Return audio URL and filename

**Code Reference**: [summarize_controller.py](app/api/controllers/summarize_controller.py#L54-L120)

---

### 5. **Audio & TTS**

#### Generate Audiobook

```http
POST /api/v1/generate-audiobook
Content-Type: application/json
```

- **Request** ([`AudiobookRequest`](app/models/schemas.py)):
  ```json
  {
    "text": "Full text to convert to speech...",
    "language": "en"
  }
  ```
- **Response** ([`AudiobookResponse`](app/models/schemas.py)):
  ```json
  {
    "audio_url": "/api/v1/audio/audio-id.wav",
    "filename": "audio-id.wav",
    "text_length": 1250,
    "language": "en",
    "message": "Audiobook generated successfully"
  }
  ```

**Code Reference**: [audiobook_controller.py](app/api/controllers/audiobook_controller.py#L10-L35)

---

#### Get Audio File

```http
GET /api/v1/audio/{filename}
```

- **Purpose**: Download generated audio
- **Response**: WAV file (audio/wav)
- **Code Reference**: [audiobook_controller.py](app/api/controllers/audiobook_controller.py#L39-L52)

---

### 6. **Transcription**

#### Transcribe Audio

```http
POST /api/v1/transcribe/
Content-Type: multipart/form-data
```

- **Fields**:
  ```
  file: MP3/WAV/MP4 file
  summarization_type: "extractive" | "abstractive"
  num_sentences: 3
  max_length: 150
  min_length: 40
  ```
- **Response** ([`TranscriptionResponse`](app/models/schemas.py)):
  ```json
  {
    "text": "Full transcription of audio...",
    "summary": "Key points from transcription...",
    "summarization_type": "extractive",
    "language": "en",
    "metadata": {
      "segments_count": 15,
      "filename": "lecture.mp3"
    }
  }
  ```

**Workflow**:

1. Validate audio file format
2. Save temporary file
3. Transcribe using Whisper (OpenAI)
4. Auto-summarize transcript
5. Clean up temp files
6. Return transcript and summary

**Code Reference**: [transcription_controller.py](app/api/controllers/transcription_controller.py#L12-L75)

---

### 7. **Chat Interface**

#### Create Workspace Chat

```http
POST /api/v1/chat/workspace
Content-Type: application/json
```

- **Request** ([`CreateWorkspaceRequest`](app/models/schemas.py)):
  ```json
  {
    "title": "My Research Assistant"
  }
  ```
- **Response** ([`CreateSessionResponse`](app/models/schemas.py)):
  ```json
  {
    "session_id": "session-uuid-123...",
    "document_id": "doc-uuid-456...",
    "created_at": "2024-07-02T10:30:00",
    "message": "Workspace chat created successfully"
  }
  ```

**Purpose**: Create interactive Q&A environment for document analysis

**Code Reference**: [chat_controller.py](app/api/controllers/chat_controller.py#L88-L110)

---

#### Chat Query (RAG)

```http
POST /api/v1/chat/session/{session_id}/query
Content-Type: application/json
```

- **Request** ([`ChatQueryRequest`](app/models/schemas.py)):
  ```json
  {
    "message": "What is the methodology used in this study?",
    "generation_type": "abstractive" // abstractive | extractive
  }
  ```
- **Response** ([`ChatQueryResponse`](app/models/schemas.py)):
  ```json
  {
    "session_id": "session-uuid-123...",
    "message": "The methodology employed in this study [1] involves a mixed-methods approach [1] combining qualitative interviews [2] with quantitative surveys [2]...",
    "citations_count": 2,
    "citations": [
      {
        "id": 1,
        "chunk_id": 15,
        "topic": "Methodology",
        "page": 8,
        "source": "research_paper.pdf",
        "text_snippet": "The study utilizes a mixed-methods approach combining...",
        "score": 0.923,
        "relevance": "high"
      },
      {
        "id": 2,
        "chunk_id": 18,
        "topic": "Research Design",
        "page": 9,
        "text_snippet": "We conducted qualitative interviews and quantitative surveys...",
        "score": 0.856,
        "relevance": "high"
      }
    ]
  }
  ```

**Workflow** (See [Chat Service Flow](#6-chat-interface-with-rag-pipeline)):

1. Search document for relevant chunks
2. Build context from top-3 results
3. Generate response using abstractive/extractive method
4. Add citation markers [1], [2], etc.
5. Save to session history
6. Return response with full citations

**Code Reference**: [chat_controller.py](app/api/controllers/chat_controller.py#L150-L220)

---

#### Get Chat History

```http
GET /api/v1/chat/session/{session_id}/history
```

- **Response**: [`ConversationHistoryResponse`](app/models/schemas.py)
  ```json
  {
    "session_id": "session-uuid-123...",
    "messages": [
      {
        "role": "user",
        "content": "What is the methodology?",
        "timestamp": "2024-07-02T10:30:00"
      },
      {
        "role": "assistant",
        "content": "The methodology employed...",
        "citations": [...]
      }
    ]
  }
  ```

**Code Reference**: [chat_controller.py](app/api/controllers/chat_controller.py#L300+)

---

### 8. **Explainable AI (XAI)**

#### Explain Extractive Summarization

```http
POST /api/v1/explain/extractive
Content-Type: application/json
```

- **Request** ([`ExplainExtractiveRequest`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "chunk_ids": [0, 1, 2],
    "num_sentences": 5,
    "generate_lrp": false // Layer-wise Relevance Propagation
  }
  ```
- **Response** ([`ExplainExtractiveResponse`](app/models/schemas.py)):
  ```json
  {
    "summary": "Selected sentence 1. Selected sentence 5. Selected sentence 12.",
    "num_sentences_input": 25,
    "num_sentences_selected": 5,
    "selected_indices": [0, 4, 11, 18, 23],
    "average_score_selected": 0.87,
    "average_score_all": 0.52,
    "score_distribution": [0.92, 0.34, 0.28, 0.95, 0.88, ...],
    "sentences": [
      {
        "index": 0,
        "text": "Selected sentence 1",
        "importance_score": 0.92,
        "is_selected": true,
        "attention_weights": [0.15, 0.82, 0.03, ...],
        "position_bias": 0.05
      }
    ],
    "explanation_methods": ["importance_scores", "attention_weights", "position_analysis"],
    "xai_type": "explainable_extraction"
  }
  ```

**Explanations Provided**:

- **Importance Scores**: Per-sentence neural network output
- **Attention Weights**: Which other sentences each sentence attended to
- **Position Bias**: How document position affects selection
- **LRP Analysis** (optional): Layer-wise relevance breakdown

**Code Reference**: [xai_controller.py](app/api/controllers/xai_controller.py#L25-L70)

---

#### Explain Search Results

```http
POST /api/v1/explain/search
Content-Type: application/json
```

- **Request** ([`ExplainSearchRequest`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "query": "What are the results?",
    "top_k": 5
  }
  ```
- **Response** ([`ExplainSearchResponse`](app/models/schemas.py)):
  ```json
  {
    "query": "What are the results?",
    "results": [
      {
        "chunk_id": 45,
        "text": "The results show...",
        "score": 0.923,
        "score_breakdown": {
          "faiss_score": 0.95,
          "faiss_weight": 0.5,
          "bm25_score": 0.88,
          "bm25_weight": 0.3,
          "tfidf_score": 0.9,
          "tfidf_weight": 0.2,
          "combined": 0.923
        },
        "ranking_reason": "High semantic similarity (0.95) combined with strong keyword match (0.88)"
      }
    ]
  }
  ```

**Explanations for Search**:

- Breaking down FAISS semantic score
- Breaking down BM25 keyword score
- Breaking down TF-IDF statistical score
- Explanation of why each result ranked where it did

**Code Reference**: [xai_controller.py](app/api/controllers/xai_controller.py#L73-L140)

---

#### Explain Abstractive Summarization

```http
POST /api/v1/explain/abstractive
Content-Type: application/json
```

- **Request** ([`ExplainAbstractiveRequest`](app/models/schemas.py)):
  ```json
  {
    "document_id": "uuid-123...",
    "chunk_ids": [0, 1, 2]
  }
  ```
- **Response**: Token-level confidence scores and attribution

**Code Reference**: [xai_controller.py](app/api/controllers/xai_controller.py#L143-L210)

---

## Core Services

### 1. Configuration Service

**File**: [app/core/config.py](app/core/config.py)

Manages all application settings:

- **Model Paths**: Extractive, Abstractive, Embeddings
- **Directory Configuration**: uploads, outputs, data directories
- **Size Limits**: PDF max size (50MB), Audio max size (10MB)
- **Default Parameters**: Sentence counts, token lengths
- **Authentication**: JWT secret, Google OAuth client ID
- **CORS Settings**: Allowed origins for cross-origin requests

```python
settings.UPLOAD_DIR        # Path to uploaded files
settings.OUTPUT_DIR        # Path to generated files
settings.DATA_DIR          # Path to chunks JSON files
settings.EXTRACTIVE_MODEL_PATH      # BiLSTM model path
settings.ABSTRACTIVE_MODEL_PATH     # T5 model path
settings.SENTENCE_ENCODER  # 'all-MiniLM-L6-v2'
settings.TTS_MODEL         # 'tts_models/multilingual/multi-dataset/xtts_v2'
```

**Reference**: [app/core/config.py](app/core/config.py#L1-L50)

---

### 2. Security & Authentication

**File**: [app/core/security.py](app/core/security.py)

Handles JWT tokens and Google OAuth:

```python
def get_current_user(token: str = Depends(HTTPBearer())) -> User:
    """
    Validates JWT token and returns current user.
    Raises 401 if token is invalid or expired.
    """
    # Decode JWT token
    # Validate signature using JWT_SECRET
    # Return User schema with user_id
```

**Reference**: [app/core/security.py](app/core/security.py)

---

### 3. Document Service

**File**: [app/services/document_service.py](app/services/document_service.py)

Core document management:

| Method                                        | Purpose                     |
| --------------------------------------------- | --------------------------- |
| `process_document(pdf_path, user_id)`         | Upload and chunk document   |
| `create_workspace_document(user_id)`          | Create empty chat workspace |
| `get_document(document_id, user_id)`          | Retrieve document metadata  |
| `get_chunks(document_id, user_id, chunk_ids)` | Get specific chunks         |
| `list_documents(user_id)`                     | List user's documents       |
| `delete_document(document_id, user_id)`       | Remove document             |
| `_load_document_chunks(document_id)`          | Load from JSON file         |
| `_save_document_chunks(document_id, chunks)`  | Persist to JSON file        |

**Key Features**:

- **Ownership Tracking**: Only users can access their documents
- **Version Control**: `.owner_registry.json` tracks document ownership
- **Lazy Loading**: Load chunks on-demand to save memory

**Reference**: [app/services/document_service.py](app/services/document_service.py#L1-L100)

---

### 4. Extractive Summarization Service

**File**: [app/services/extractive_service.py](app/services/extractive_service.py)

Neural extractive summarization using BiLSTM + Attention:

**Model Architecture**:

```
Input (384-dim sentence embeddings)
    ↓
Positional Encoding (learned embeddings for sentence position)
    ↓
Bidirectional LSTM (256 hidden, 2 layers)
    └─→ Captures context and dependencies between sentences
    ↓
Layer Normalization
    ↓
Multi-Head Attention (8 heads, 256*2 dim)
    └─→ Learns which sentences are important
    ↓
Layer Normalization
    ↓
Feed-Forward Classifier (256 → 128 → 1)
    └─→ Outputs importance score [0, 1] for each sentence
    ↓
Output: Sentence importance scores
```

**Key Methods**:

```python
ExtractiveSummarizer.summarize(
    text: str,
    num_sentences: int = 5
) -> str:
    """Select top-N important sentences from text"""
    # 1. Tokenize into sentences
    # 2. Encode sentences with SentenceTransformer
    # 3. Run through BiLSTM model
    # 4. Select top-N by score
    # 5. Return sentences in original order
```

**Reference**: [app/services/extractive_service.py](app/services/extractive_service.py#L1-L220)

---

### 5. Abstractive Summarization Service

**File**: [app/services/abstractive_service.py](app/services/abstractive_service.py)

Fine-tuned T5 for text generation:

**Key Methods**:

```python
AbstractiveSummarizer.summarize(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 8
) -> str:
    """Generate concise summary using T5 transformer"""
    # 1. Prepend "summarize: " to text
    # 2. Tokenize and truncate to 512 tokens
    # 3. Generate using beam search with 8 beams
    # 4. Apply diversity and repetition penalties
    # 5. Decode and return summary text
```

**Generation Parameters**:

- **num_beams**: 8 - Explore 8 most likely sequences
- **length_penalty**: 2.0 - Prefer shorter summaries
- **repetition_penalty**: 3.0 - Discourage repeating words
- **diversity_penalty**: 1.5 - Encourage diverse beams
- **no_repeat_ngram_size**: 3 - Prevent 3-gram repetition

**Reference**: [app/services/abstractive_service.py](app/services/abstractive_service.py#L1-L65)

---

### 6. Vector Store & Hybrid Search Service

**File**: [app/services/vector_store_service.py](app/services/vector_store_service.py)

Multi-algorithm search combining semantic, keyword, and statistical methods:

**Components**:

```python
HybridVectorStore:
    ├─ FAISS Index
    │  └─ SentenceTransformer embeddings (384-dim)
    │     └─ L2 distance-based semantic search
    ├─ BM25 Index
    │  └─ Okapi BM25 algorithm
    │     └─ Keyword-based ranking
    └─ TF-IDF Matrix
       └─ Term frequency-inverse document frequency
          └─ Statistical ranking
```

**Search Methods**:

```python
HybridVectorStore.search(
    query: str,
    top_k: int = 5,
    faiss_weight: 0.5,
    bm25_weight: 0.3,
    tfidf_weight: 0.2
) -> List[Dict]:
    """
    1. Encode query with SentenceTransformer
    2. Search FAISS index, get distances → normalize to [0,1]
    3. Search BM25 index, get scores → normalize to [0,1]
    4. Search TF-IDF matrix, get similarity → normalize to [0,1]
    5. Combine scores: final = 0.5*faiss + 0.3*bm25 + 0.2*tfidf
    6. Return top-K by combined score
    """
```

**Reference**: [app/services/vector_store_service.py](app/services/vector_store_service.py#L1-L150)

---

### 7. Chat (RAG) Service

**File**: [app/services/chat_service.py](app/services/chat_service.py)

Retrieval-Augmented Generation for question answering:

**Key Methods**:

```python
RAGChatService.generate_response(
    query: str,
    retrieved_chunks: List[Dict],
    generation_type: str = "abstractive"
) -> Tuple[str, List[Dict]]:
    """
    1. Build context from top-3 chunks
    2. Generate response using abstractive/extractive method
    3. Create citations from source chunks
    4. Add citation markers [1], [2], [3] to response
    5. Return response + citations metadata
    """
```

**Reference**: [app/services/chat_service.py](app/services/chat_service.py#L1-L100)

---

### 8. Session Management Service

**File**: [app/services/session_service.py](app/services/session_service.py)

Manages chat sessions and conversation history:

**Key Methods**:

```python
SessionManager.create_session(user_id, document_id, metadata) → session_id
SessionManager.get_session(session_id, user_id) → Session
SessionManager.add_message(session_id, user_id, role, content, citations)
SessionManager.add_source(session_id, user_id, source_metadata)
SessionManager.list_sessions(user_id) → List[Session]
```

**Session Structure**:

```json
{
  "session_id": "uuid",
  "document_id": "uuid",
  "user_id": "google_user_id",
  "created_at": "2024-07-02T10:30:00",
  "updated_at": "2024-07-02T10:35:00",
  "messages": [
    {
      "role": "user|assistant",
      "content": "message text",
      "timestamp": "2024-07-02T10:30:05",
      "citations": [...]
    }
  ],
  "metadata": {
    "title": "Chat title",
    "sources": [...]
  }
}
```

**Storage**: `data/sessions/{session_id}.json`

**Reference**: [app/services/session_service.py](app/services/session_service.py#L1-L100)

---

### 9. Text-to-Speech Service

**File**: [app/services/tts_service.py](app/services/tts_service.py)

Converts text to natural-sounding speech:

```python
AudiobookGenerator.generate(
    text: str,
    output_path: str,
    speaker_wav: Optional[str] = None,
    language: str = "en"
) -> str:
    """
    Uses Coqui TTS (XTTS V2) for multilingual synthesis.
    - If speaker_wav provided: Clone voice style
    - Else: Use default reference audio
    - Output: WAV file at output_path
    """
```

**Supported Languages**: en, es, fr, de, it, ja, zh, etc. (multilingual)

**Reference**: [app/services/tts_service.py](app/services/tts_service.py#L1-L60)

---

### 10. Transcription Service

**File**: [app/services/transcription_service.py](app/services/transcription_service.py)

Converts audio/video to text using Whisper:

```python
TranscriptionService.transcribe(audio_path: str) -> Dict:
    """
    Uses OpenAI Whisper model for speech recognition.
    Returns: {
        "text": "full transcription...",
        "segments": [...],  # with timestamps
        "language": "en"
    }
    """
```

**Model**: Whisper 'base' (configurable)

---

### 11. XAI (Explainable AI) Service

**File**: [app/services/xai_service.py](app/services/xai_service.py)

Provides insights into model decisions:

**Classes**:

- `ExplainableExtractiveService` - Explain sentence selection
- `ExplainableSearchService` - Explain search ranking
- `ExplainableAbstractiveService` - Explain token generation

**Explanation Methods**:

- **Attention Visualization**: Show attention weights between sentences
- **Importance Scores**: Per-sentence neural network outputs
- **Relevance Propagation (LRP)**: Layer-wise contribution analysis
- **SHAP Values**: Shapley additive explanations
- **LIME**: Local interpretable model-agnostic explanations

**Reference**: [app/services/xai_service.py](app/services/xai_service.py)

---

### 12. PDF/Document Processing Utility

**File**: [app/utils/pdf_processor.py](app/utils/pdf_processor.py)

Robust document extraction and chunking:

**DocumentProcessor Methods**:

```python
DocumentProcessor.process_file(file_path) -> List[Dict]:
    """Route to specific format handler based on extension"""
    # .pdf → process_pdf()
    # .pptx → process_pptx()
    # .docx → process_docx()
    # .txt/.md/.csv → process_text_file()

DocumentProcessor.process_pdf(pdf_path) -> List[Dict]:
    """
    1. Open PDF with PyMuPDF (fitz)
    2. Analyze body font size for header detection
    3. Extract text with formatting metadata
    4. Generate captions for embedded images (BLIP)
    5. Create semantic chunks with topics
    """
```

**TextPreprocessor**:

```python
TextPreprocessor.preprocess_chunks(chunks) -> List[Dict]:
    """
    1. Clean text (remove extra whitespace)
    2. Tokenize into sentences
    3. Extract keywords
    4. Generate topic summaries
    5. Add metadata (page, topic, chunk_id)
    """
```

**Reference**: [app/utils/pdf_processor.py](app/utils/pdf_processor.py#L1-L100)

---

## Frontend Structure

**Framework**: React 19 + Vite + Tailwind CSS

**Directory Structure**:

```
frontend/
├── src/
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # React entry point
│   ├── index.css            # Tailwind styles
│   ├── api/                 # API client functions
│   │   └── apiClient.js     # Axios/Fetch wrapper
│   ├── components/          # UI components
│   │   ├── Header.jsx       # Navigation & auth
│   │   ├── DocumentUpload.jsx
│   │   ├── ChatInterface.jsx
│   │   ├── SearchBar.jsx
│   │   ├── SummaryView.jsx
│   │   └── AudioPlayer.jsx
│   └── store/               # State management
│       └── useStore.js      # Zustand/Context store
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
└── index.html               # HTML entry point
```

**Key Features**:

- OAuth 2.0 Google login
- Document upload with drag-and-drop
- Real-time search and retrieval
- Chat interface with citations
- Audio playback for generated audiobooks
- Responsive design

---

## Data Storage & Management

### Directory Structure

```
auralearn/
├── uploads/                 # User-uploaded files
│   ├── {doc_id}_filename.pdf
│   ├── {doc_id}_document.docx
│   └── temp_audio_{uuid}.wav
│
├── outputs/                 # Generated files
│   ├── {audio_id}.wav       # Generated audiobooks
│   └── {audio_id}_speakers.json
│
├── data/
│   ├── {document_id}_chunks.json  # Document chunks
│   ├── .owner_registry.json       # Document ownership
│   └── sessions/
│       └── {session_id}.json      # Chat history
│
├── models/
│   ├── extractive_model_final.pt  # BiLSTM model
│   └── t5_summarizer/             # T5 model directory
│
└── __pycache__/             # Compiled Python files
```

---

### Data Formats

#### Document Chunks JSON

**File**: `data/{document_id}_chunks.json`

```json
[
  {
    "chunk_id": 0,
    "topic": "Introduction",
    "page": 1,
    "text": "This research investigates...",
    "text_length": 145,
    "sentences": ["Sentence 1", "Sentence 2"],
    "keywords": ["research", "investigation", "methodology"],
    "timestamp": "2024-07-02T10:30:00"
  },
  {
    "chunk_id": 1,
    "topic": "Literature Review",
    "page": 2,
    "text": "Previous studies have shown...",
    "text_length": 189,
    "sentences": ["Sentence 1", "Sentence 2"],
    "keywords": ["studies", "research", "findings"],
    "timestamp": "2024-07-02T10:30:00"
  }
]
```

---

#### Session History JSON

**File**: `data/sessions/{session_id}.json`

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "12345678-1234-1234-1234-123456789012",
  "user_id": "google_user_id_12345",
  "created_at": "2024-07-02T10:30:00",
  "updated_at": "2024-07-02T10:35:45",
  "messages": [
    {
      "role": "user",
      "content": "What is the main conclusion?",
      "timestamp": "2024-07-02T10:30:05",
      "citations": []
    },
    {
      "role": "assistant",
      "content": "The main conclusion [1] is that this approach [2] is more efficient...",
      "timestamp": "2024-07-02T10:30:10",
      "citations": [
        {
          "id": 1,
          "chunk_id": 42,
          "topic": "Conclusion",
          "page": 15,
          "text_snippet": "The main conclusion of our research...",
          "score": 0.94,
          "relevance": "high"
        },
        {
          "id": 2,
          "chunk_id": 45,
          "topic": "Efficiency",
          "page": 16,
          "text_snippet": "This approach is more efficient than...",
          "score": 0.87,
          "relevance": "high"
        }
      ]
    }
  ],
  "metadata": {
    "title": "Research Analysis Chat",
    "sources": []
  }
}
```

---

#### Owner Registry JSON

**File**: `data/.owner_registry.json`

```json
{
  "doc-uuid-1": "google_user_id_123",
  "doc-uuid-2": "google_user_id_456",
  "doc-uuid-3": "google_user_id_123"
}
```

---

## Complete User Workflows

### Workflow 1: Upload & Quick Summary

```
User Action                    Backend Processing              Storage/Output
──────────────────────────────────────────────────────────────────────────────

1. Open Frontend
   └─→ OAuth Google Login
       └─→ Get JWT token                                 [JWT in localStorage]

2. Click "Upload PDF"
   └─→ Select research_paper.pdf
       ├─→ POST /api/v1/upload
       │   ├─→ Validate file type & size
       │   ├─→ Save to uploads/{doc_id}_research_paper.pdf
       │   ├─→ Extract text → 50 chunks
       │   ├─→ Save to data/{doc_id}_chunks.json
       │   ├─→ Build vector indexes (FAISS, BM25, TF-IDF)
       │   └─→ Store ownership in .owner_registry.json
       │
       └─→ Return UploadPDFResponse
           ├─→ document_id: "uuid-123"
           ├─→ num_chunks: 50
           └─→ Display in UI

3. User clicks "Summarize"
   └─→ POST /api/v1/summarize
       ├─→ Load chunks from data/{doc_id}_chunks.json
       ├─→ Combine all chunk texts
       ├─→ Run through BiLSTM + Attention
       ├─→ Select top 5 sentences by importance
       └─→ Return SummarizeResponse
           ├─→ summary: "Sentence 1. Sentence 5. Sentence 12..."
           └─→ Display in UI

4. User clicks "Generate Audio"
   └─→ POST /api/v1/generate-audiobook
       ├─→ Take summary text
       ├─→ Coqui TTS converts to speech
       ├─→ Save to outputs/{audio_id}.wav
       └─→ Return audio URL
           ├─→ /api/v1/audio/{audio_id}.wav
           └─→ Play in browser
```

**Time to completion**: ~5-10 seconds

---

### Workflow 2: Search & Chat

```
User Action                    Backend Processing              Output
──────────────────────────────────────────────────────────────────────────────

1. Create Chat
   └─→ POST /api/v1/chat/workspace
       ├─→ Create new blank document
       ├─→ Create session
       └─→ Save to data/sessions/{session_id}.json

2. Upload Documents to Chat
   └─→ POST /api/v1/chat/session/{session_id}/upload-source
       ├─→ Save source files to chat document
       ├─→ Save metadata in session
       └─→ Return source list

3. Ask Question
   ├─→ User: "What methodology was used?"
   │
   └─→ POST /api/v1/chat/session/{session_id}/query
       ├─→ [Vector Store Search]
       │   ├─→ Encode query with SentenceTransformer
       │   ├─→ FAISS search → top 10 candidates
       │   ├─→ BM25 search → scores on all chunks
       │   ├─→ TF-IDF search → similarity on all chunks
       │   ├─→ Combine: 0.5*FAISS + 0.3*BM25 + 0.2*TF-IDF
       │   └─→ Return top-3 results
       │
       ├─→ [RAG Response Generation]
       │   ├─→ Build context from top-3 chunks
       │   ├─→ Generate response using abstractive T5
       │   │   "The methodology employed in this study
       │   │    involves a mixed-methods approach [1]
       │   │    combining qualitative interviews [2]..."
       │   ├─→ Create citations for each source
       │   └─→ Add [1], [2] markers to response
       │
       ├─→ [Save to Session]
       │   ├─→ Store user message
       │   ├─→ Store assistant response + citations
       │   └─→ Update data/sessions/{session_id}.json
       │
       └─→ Return ChatQueryResponse
           ├─→ Response text with citations
           ├─→ Citations: [Chunk 15 from page 8, Chunk 18 from page 9]
           └─→ Display in chat with clickable references

4. View Chat History
   └─→ GET /api/v1/chat/session/{session_id}/history
       ├─→ Load data/sessions/{session_id}.json
       ├─→ Return all messages with citations
       └─→ Display conversation thread
```

**Latency breakdown**:

- Search: 100-200ms
- T5 generation: 2-5 seconds
- Total: ~2-5 seconds per message

---

### Workflow 3: Transcribe Audio

```
User Action                    Backend Processing              Storage/Output
──────────────────────────────────────────────────────────────────────────────

1. Click "Upload Audio"
   └─→ Select lecture.mp3 (60 minutes)

2. Upload
   └─→ POST /api/v1/transcribe/
       ├─→ Save temp file: uploads/{uuid}.mp3
       │
       ├─→ [Transcription]
       │   ├─→ Load Whisper 'base' model
       │   ├─→ Decode MP3 audio
       │   ├─→ Process in 30-second chunks
       │   ├─→ Whisper processes audio with attention
       │   ├─→ Generate transcript with timestamps
       │   └─→ Detect language (auto)
       │
       ├─→ [Auto-summarization]
       │   ├─→ Select Extractive summarization
       │   ├─→ BiLSTM selects top 5 key sentences
       │   └─→ Summary: "First key point. Second key point..."
       │
       ├─→ Clean up temp file
       │
       └─→ Return TranscriptionResponse
           ├─→ Full transcript (3000 words)
           ├─→ Summary (50 words)
           ├─→ Language detected: "en"
           └─→ Display both in UI

3. User saves to document
   └─→ Save transcript as document
       ├─→ Create new document
       ├─→ Process transcript into chunks
       ├─→ Build indexes
       └─→ Can now summarize or chat about it
```

**Processing time**: ~2-3x audio length (CPU dependent)

---

### Workflow 4: Explainable AI

```
User Action                    Model Processing               Output
──────────────────────────────────────────────────────────────────────────────

1. Summarize and request explanation
   └─→ POST /api/v1/explain/extractive
       ├─→ Load original text and selected sentences
       ├─→ [BiLSTM Model Analysis]
       │   ├─→ Forward pass through model
       │   ├─→ Extract attention weights
       │   │   └─→ [8 heads × 256 dim] → show which
       │   │       sentences attended to others
       │   ├─→ Extract importance scores
       │   │   └─→ Why each sentence scored 0-1
       │   ├─→ Positional analysis
       │   │   └─→ How position affected selection
       │   └─→ Optional: Layer-wise Relevance Propagation
       │       └─→ Trace contribution through network
       │
       └─→ Return ExplainExtractiveResponse
           ├─→ Summary text
           ├─→ Importance scores per sentence
           ├─→ Attention weight visualization
           ├─→ Position bias analysis
           └─→ Display in UI with interactive visualization

2. Search and request explanation
   └─→ POST /api/v1/explain/search
       ├─→ Show score breakdown:
       │   ├─→ FAISS (semantic): 0.95
       │   │   └─→ "High semantic similarity to query"
       │   ├─→ BM25 (keyword): 0.88
       │   │   └─→ "Exact match on 'methodology'"
       │   └─→ TF-IDF (statistical): 0.90
       │       └─→ "High term frequency of query words"
       │
       └─→ Combined score: 0.923
           └─→ "Ranked #1 due to strong semantic
               match combined with keyword presence"
```

---

## Application Startup Flow

```python
# main.py entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7533,
        reload=True
    )
```

**Startup Sequence**:

```
1. FastAPI App instantiation
   └─→ @asynccontextmanager async def lifespan()

2. Lifespan - Startup Phase
   └─→ print("Initializing services...")
       ├─→ [service_registry.init_services()]
       │   ├─→ Load extractive model
       │   │   └─→ models/extractive_model_final.pt
       │   ├─→ Load abstractive model
       │   │   └─→ models/t5_summarizer/
       │   ├─→ Load sentence encoder
       │   │   └─→ Download 'all-MiniLM-L6-v2' from HuggingFace
       │   ├─→ Initialize TTS engine
       │   │   └─→ Load 'tts_models/multilingual/multi-dataset/xtts_v2'
       │   ├─→ Initialize document manager
       │   ├─→ Initialize session manager
       │   └─→ Initialize vector store cache
       │
       └─→ print("Services initialized successfully")

3. CORS Middleware Configuration
   └─→ Allow cross-origin requests from frontend

4. Include API Routers
   ├─→ /api/v1/upload
   ├─→ /api/v1/document
   ├─→ /api/v1/summarize
   ├─→ /api/v1/search
   ├─→ /api/v1/chat
   ├─→ /api/v1/generate-audiobook
   ├─→ /api/v1/transcribe
   ├─→ /api/v1/explain
   └─→ /api/v1/health

5. Server listening on http://0.0.0.0:7533

6. Lifespan - Shutdown Phase (on app close)
   └─→ print("Shutting down...")
       ├─→ Unload models
       ├─→ Close file handles
       └─→ Clean temp files
```

**Reference**: [main.py](main.py#L1-L50)

---

## Key Technologies & Why They're Used

| Technology               | Purpose                   | Why Chosen                                        |
| ------------------------ | ------------------------- | ------------------------------------------------- |
| **FastAPI**              | Web framework             | Modern, automatic validation, fast                |
| **PyMuPDF**              | PDF extraction            | Robust, fast, accurate text/image extraction      |
| **SentenceTransformers** | Text embeddings           | Small, efficient, high-quality 384-dim embeddings |
| **FAISS**                | Vector search             | Sub-millisecond search, GPU-compatible            |
| **BM25**                 | Keyword search            | Industry-standard, proven algorithm               |
| **BiLSTM + Attention**   | Extractive summarization  | Captures dependencies, lightweight                |
| **T5**                   | Abstractive summarization | State-of-the-art text generation, fine-tunable    |
| **Coqui TTS**            | Text-to-speech            | Multilingual, open-source, high quality           |
| **Whisper**              | Audio transcription       | Accurate, multilingual, robust                    |
| **React**                | Frontend                  | Component-based, reactive updates                 |
| **Tailwind CSS**         | Styling                   | Utility-first, responsive design                  |

---

## Performance Characteristics

### Latency (Single Request)

| Operation                             | Time      | Notes                                 |
| ------------------------------------- | --------- | ------------------------------------- |
| Document Upload (50 MB PDF)           | 30-60s    | Text extraction + indexing            |
| Search Query                          | 100-200ms | Hybrid search (FAISS + BM25 + TF-IDF) |
| Extractive Summarization (500 words)  | 200-400ms | BiLSTM inference                      |
| Abstractive Summarization (500 words) | 2-5s      | T5 beam search generation             |
| TTS (100 words)                       | 5-10s     | Mel-spectrogram + vocoder             |
| Audio Transcription (5 min)           | 10-30s    | Whisper model inference               |
| Search + Generate Response            | 2-5s      | Search + RAG response generation      |

### Memory Usage

| Component                 | Memory   |
| ------------------------- | -------- |
| Extractive Model          | ~150MB   |
| Abstractive Model (T5)    | ~400MB   |
| SentenceTransformer       | ~50MB    |
| FAISS Index (1000 chunks) | ~2MB     |
| TTS Model (XTTS V2)       | ~2GB     |
| Python Runtime            | ~200MB   |
| **Total**                 | **~3GB** |

### Scalability

- **Concurrent Users**: 10-50 (depending on hardware)
- **Documents per User**: Unlimited (storage dependent)
- **Chunks per Document**: 1,000+ (search latency increases with size)
- **Vector Index Size**: FAISS handles millions of vectors efficiently

---

## Error Handling Strategy

```python
# Controllers implement try-except with proper HTTP status codes

@router.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    try:
        # Get chunks
        chunks = document_manager.get_chunks(...)
        if chunks is None:
            raise HTTPException(404, "Document not found")

        # Summarize
        summary = summarizer.summarize(...)

        return SummarizeResponse(...)

    except HTTPException:
        raise  # Re-raise expected errors
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(500, f"Summarization failed: {str(e)}")
```

**Status Codes Used**:

- **200**: Success
- **400**: Bad request (invalid parameters)
- **401**: Unauthorized (invalid token)
- **403**: Forbidden (no access)
- **404**: Not found (document doesn't exist)
- **500**: Server error (model crash, etc.)
- **503**: Service unavailable (model not loaded)

---

## Configuration & Environment Variables

**File**: `.env` (loaded by Pydantic)

```bash
GOOGLE_CLIENT_ID=your-google-client-id
JWT_SECRET=your-secret-key
EXTRACTIVE_MODEL_PATH=models/extractive_model_final.pt
ABSTRACTIVE_MODEL_PATH=models/t5_summarizer
SENTENCE_ENCODER=all-MiniLM-L6-v2
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
WHISPER_MODEL_SIZE=base
MAX_PDF_SIZE=52428800  # 50MB
DEFAULT_EXTRACTIVE_SENTENCES=5
DEFAULT_ABSTRACTIVE_MAX_LENGTH=150
DEFAULT_ABSTRACTIVE_MIN_LENGTH=40
ALLOWED_ORIGINS=*
```

**Reference**: [app/core/config.py](app/core/config.py#L20-L40)

---

## Summary

**AuraLearn** is a sophisticated document intelligence platform combining:

1. **Deep Learning**: BiLSTM extractive, T5 abstractive, Transformers for embeddings
2. **Information Retrieval**: Hybrid search (semantic + keyword + statistical)
3. **Speech Integration**: TTS for audiobooks, Whisper for transcription
4. **Chat Interface**: RAG (Retrieval-Augmented Generation) with citations
5. **Explainability**: Visualize model decisions and search rankings
6. **User Management**: OAuth-based authentication with per-user data isolation

**Technology Stack**: FastAPI (backend) + React (frontend) + PyTorch (ML) + FAISS (search)

All components are modular, scalable, and designed for production use.

---

## Quick Reference Links

### Key Entry Points

- **Server Entry**: [main.py](main.py)
- **API Routes**: [app/api/controllers/**init**.py](app/api/controllers/__init__.py)
- **Configuration**: [app/core/config.py](app/core/config.py)

### Controllers (API Endpoints)

- Upload: [upload_controller.py](app/api/controllers/upload_controller.py)
- Documents: [document_controller.py](app/api/controllers/document_controller.py)
- Search: [search_controller.py](app/api/controllers/search_controller.py)
- Summarize: [summarize_controller.py](app/api/controllers/summarize_controller.py)
- Chat: [chat_controller.py](app/api/controllers/chat_controller.py)
- Audio: [audiobook_controller.py](app/api/controllers/audiobook_controller.py)
- Transcription: [transcription_controller.py](app/api/controllers/transcription_controller.py)
- Explain: [xai_controller.py](app/api/controllers/xai_controller.py)

### Services (Business Logic)

- Documents: [document_service.py](app/services/document_service.py)
- Extractive: [extractive_service.py](app/services/extractive_service.py)
- Abstractive: [abstractive_service.py](app/services/abstractive_service.py)
- Vector Store: [vector_store_service.py](app/services/vector_store_service.py)
- Chat: [chat_service.py](app/services/chat_service.py)
- Sessions: [session_service.py](app/services/session_service.py)
- TTS: [tts_service.py](app/services/tts_service.py)
- XAI: [xai_service.py](app/services/xai_service.py)

### Utilities

- PDF Processor: [pdf_processor.py](app/utils/pdf_processor.py)
- Schemas: [schemas.py](app/models/schemas.py)
- Security: [security.py](app/core/security.py)

---

**Created**: July 2, 2026  
**Version**: 1.0.0  
**Status**: Production Ready
