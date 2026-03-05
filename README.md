<img src="./assets/banner.png" >

### AuraLearn API

Professional document intelligence API for PDF processing, summarization, and audio generation.

### Key capabilities

- **PDF ingestion** with robust chunking and metadata extraction
- **Extractive summarization** using a BiLSTM-based model
- **Abstractive summarization** using a fine‑tuned T5 model
- **Audiobook generation** via high-quality TTS
- **End-to-end pipeline** from document upload to audio output

### Architecture
```mermaid
---
config:
  theme: default
  themeVariables:
    fontSize: 48px
  layout: dagre
  look: classic
---
flowchart TD

A[PDF Upload] --> B[PDF Parsing & Metadata Extraction]
B --> C[Intelligent Chunking]
C --> D[Chunk Storage with Metadata]

D --> E[SentenceTransformer Embedding]
E --> F[FAISS Vector Indexing]

D --> G[BM25 Index]
D --> H[TF-IDF Matrix]

I[User Query] --> J[Query Embedding]
J --> K[FAISS Semantic Retrieval]
I --> L[BM25 Retrieval]
I --> M[TF-IDF Retrieval]

K --> N[Hybrid Score Fusion]
L --> N
M --> N

N --> O{Top-K Chunks Selected}

O --> P[ExtractiveModel<br/>BiLSTM + Multi-Head Attention]
P --> Q[Sentence Importance Scores]
Q --> R{Top Sentences Filtered}

R --> S[T5 Transformer<br/>Abstractive Synthesis]
S --> T[Final Summary Text]

T --> U[TTS Engine]
U --> V[Audio Output]

%% XAI Layer
N --> X1[Search Explanations]
P --> X2[Extractive Attention Visualization]
S --> X3[Abstractive Token Confidence]
Q --> X4[Leave-One-Out Sensitivity]

X1 --> Z[XAI Dashboard]
X2 --> Z
X3 --> Z
X4 --> Z

Z -. Feedback Loop .-> P
Z -. Feedback Loop .-> S
Z -. Feedback Loop .-> N

%% Styling XAI Components
classDef xai fill:#FFE8B3,stroke:#D97706,stroke-width:3px,color:#000000;
class X1,X2,X3,X4,Z xai;
linkStyle default stroke-width:4px;
```
