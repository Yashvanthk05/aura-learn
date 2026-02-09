import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class HybridVectorStore:

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        self.tokenized_corpus = []
        
    def create_index(self, chunks: List[Dict]):
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        if not texts:
            print("WARN: No text chunks to index.")
            return
        
        print(f"INFO: Creating hybrid index for {len(texts)} chunks...")
    
        print("Creating FAISS index...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        print("Creating BM25 index...")
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print("Creating TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"INFO: Hybrid index created successfully")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        faiss_weight: float = 0.5,
        bm25_weight: float = 0.3,
        tfidf_weight: float = 0.2
    ) -> List[Dict]:
        
        if not self.chunks:
            return []
        
        total_weight = faiss_weight + bm25_weight + tfidf_weight
        faiss_weight /= total_weight
        bm25_weight /= total_weight
        tfidf_weight /= total_weight
        
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        k_candidates = min(top_k * 3, len(self.chunks))
        distances, indices = self.faiss_index.search(query_embedding, k_candidates)
        
        max_distance = distances[0].max() if distances[0].max() > 0 else 1.0
        faiss_scores = 1 - (distances[0] / max_distance)

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_scores = bm25_scores / max_bm25
        
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_tfidf.T).toarray().flatten()
        
        max_tfidf = tfidf_scores.max() if tfidf_scores.max() > 0 else 1.0
        tfidf_scores = tfidf_scores / max_tfidf
        
        combined_scores = {}
        
        for idx, faiss_idx in enumerate(indices[0]):
            combined_scores[faiss_idx] = faiss_scores[idx] * faiss_weight
        
        for idx in range(len(self.chunks)):
            if idx not in combined_scores:
                combined_scores[idx] = 0
            combined_scores[idx] += bm25_scores[idx] * bm25_weight
            combined_scores[idx] += tfidf_scores[idx] * tfidf_weight
        
        sorted_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for idx, combined_score in sorted_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(combined_score)
            chunk['score_breakdown'] = {
                'faiss': float(faiss_scores[np.where(indices[0] == idx)[0][0]]) * faiss_weight if idx in indices[0] else 0,
                'bm25': float(bm25_scores[idx]) * bm25_weight,
                'tfidf': float(tfidf_scores[idx]) * tfidf_weight,
                'combined': float(combined_score)
            }
            results.append(chunk)
        
        return results
    
    def search_faiss_only(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.faiss_index:
            return []
        
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(1 - distance / (distances[0].max() + 1e-6))
            chunk['distance'] = float(distance)
            results.append(chunk)
        
        return results
    
    def search_bm25_only(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(scores[idx])
            results.append(chunk)
        
        return results
    
    def save_index(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(directory / "faiss.index"))
        
        with open(directory / "hybrid_data.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'tokenized_corpus': self.tokenized_corpus,
                'bm25': self.bm25,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)
        
        print(f"INFO: Hybrid index saved to {directory}")
    
    def load_index(self, directory: Path):
        directory = Path(directory)
        
        faiss_path = directory / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        data_path = directory / "hybrid_data.pkl"
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.tokenized_corpus = data['tokenized_corpus']
                self.bm25 = data['bm25']
                self.tfidf_vectorizer = data['tfidf_vectorizer']
                self.tfidf_matrix = data['tfidf_matrix']
        
        print(f"INFO: Hybrid index loaded from {directory}")
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        for chunk in self.chunks:
            if chunk.get('chunk_id') == chunk_id:
                return chunk
        return None
    
    def get_statistics(self) -> Dict:
        return {
            'total_chunks': len(self.chunks),
            'faiss_indexed': self.faiss_index.ntotal if self.faiss_index else 0,
            'bm25_indexed': len(self.tokenized_corpus),
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'model_name': self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else 'unknown'
        }
