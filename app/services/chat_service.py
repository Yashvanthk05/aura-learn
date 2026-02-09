from typing import List, Dict, Optional, Tuple
import re
from app.services.extractive_service import ExtractiveSummarizer
from app.core.config import settings

class RAGChatService:
    
    def __init__(
        self,
        model_name: str = settings.EXTRACTIVE_MODEL_PATH,
        device: Optional[str] = None
    ):
        self.device = device if device else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading extractive summarizer model: {model_name}...")
        
        try:
            self.generator = ExtractiveSummarizer(
                model_path=model_name,
                encoder_name='all-MiniLM-L6-v2'
            )
        except Exception as e:
            print(f"WARN: Could not load {model_name}, using fallback simple generation: {e}")
            self.generator = None
        
        print(f"Chat model loaded on {self.device}")
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        conversation_history: Optional[str] = None,
        max_context_chunks: int = 3
    ) -> Tuple[str, List[Dict]]:
        
        context_chunks = retrieved_chunks[:max_context_chunks]
        citations = self._build_citations(context_chunks)
        context = self._build_context(context_chunks)
        
        if self.generator and context.strip():
            response = self._generate_with_model(context)
        else:
            response = self._generate_extractive_response(query, context_chunks)
        
        response_with_citations = self._add_citation_markers(response, citations)
        return response_with_citations, citations
    
    def _build_citations(self, chunks: List[Dict]) -> List[Dict]:
        citations = []
        
        for idx, chunk in enumerate(chunks):
            citation = {
                'id': idx + 1,
                'chunk_id': chunk.get('chunk_id', 'unknown'),
                'topic': chunk.get('topic', 'Unknown'),
                'page': chunk.get('page', 'N/A'),
                'source': chunk.get('source_name', 'Document'),
                'text_snippet': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'score': chunk.get('score', 0.0),
                'relevance': 'high' if chunk.get('score', 0) > 0.7 else 'medium' if chunk.get('score', 0) > 0.4 else 'low'
            }
            citations.append(citation)
        
        return citations
    
    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        
        for idx, chunk in enumerate(chunks):
            topic = chunk.get('topic', 'Content')
            text = chunk['text']
            context_parts.append(f"[Context {idx + 1}] {topic}: {text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_with_model(self, context: str) -> str:

        try:
            response = self.generator.summarize(
                context,
                num_sentences=3,
                min_confidence=0.0,
                diverse_selection=True,
                diversity_penalty=0.3,
                preserve_order=True
            )
            return response if response.strip() else "I don't have enough information in the document to answer that question."
        except Exception as e:
            print(f"WARN: Generation failed: {e}, using extractive fallback")
            return "I don't have enough information in the document to answer that question."
    
    def _generate_extractive_response(
        self,
        query: str,
        chunks: List[Dict]
    ) -> str:
        
        if not chunks:
            return "I don't have enough information in the document to answer that question."
        
        # Get the most relevant chunk
        top_chunk = chunks[0]
        text = top_chunk['text']
        
        # Simple extractive response
        sentences = text.split('.')
        
        # Return first 2-3 sentences as response
        response_sentences = sentences[:min(3, len(sentences))]
        response = '. '.join(s.strip() for s in response_sentences if s.strip())
        
        if response and not response.endswith('.'):
            response += '.'
        
        # Add context about the source
        topic = top_chunk.get('topic', 'the document')
        return f"Based on the section '{topic}': {response}"
    
    def _add_citation_markers(self, response: str, citations: List[Dict]) -> str:
        
        if not citations:
            return response
        
        citation_markers = ', '.join([f"[{c['id']}]" for c in citations])
        
        if not re.search(r'\[\d+\]', response):
            response = f"{response} {citation_markers}"
        
        return response
    
    def summarize_conversation(self, messages: List[Dict]) -> str:
        
        if not messages:
            return ""
        
        topics = []
        for msg in messages:
            if msg['role'] == 'user':
                words = msg['content'].split()
                key_words = [w for w in words if len(w) > 4][:3]
                topics.extend(key_words)
        
        unique_topics = list(dict.fromkeys(topics))[:5]
        
        return f"Discussed topics: {', '.join(unique_topics)}"
    
    def check_model_loaded(self) -> bool:
        return self.generator is not None
    
    def get_model_info(self) -> Dict:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.generator is not None
        }