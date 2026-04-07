"""
Document management service.
"""
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from app.utils.pdf_processor import DocumentProcessor, TextPreprocessor

class DocumentManager:
    
    def __init__(self, data_dir: Path):
       
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.doc_processor = DocumentProcessor()
        self.preprocessor = TextPreprocessor()
        self.registry_file = self.data_dir / ".owner_registry.json"
        
    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
        
    def _save_registry(self, registry: Dict):
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    @staticmethod
    def _extract_owner_id(registry_entry) -> Optional[str]:
        if isinstance(registry_entry, dict):
            return registry_entry.get("owner") or registry_entry.get("user_id")
        if isinstance(registry_entry, str):
            return registry_entry
        return None

    def _is_owned_by_user(self, document_id: str, user_id: str) -> bool:
        registry = self._load_registry()
        owner_id = self._extract_owner_id(registry.get(document_id))
        return owner_id == user_id

    def _set_owner(self, document_id: str, user_id: str):
        registry = self._load_registry()
        registry[document_id] = user_id
        self._save_registry(registry)
    
    def process_document(self, pdf_path: str, user_id: str, document_id: Optional[str] = None) -> Dict:

        if document_id is None:
            document_id = str(uuid.uuid4())
        
        processed_chunks = self.process_file_to_chunks(pdf_path)
        
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2)
            
        self._set_owner(document_id, user_id)
        
        return {
            "document_id": document_id,
            "filename": Path(pdf_path).name,
            "num_chunks": len(processed_chunks),
            "chunks": processed_chunks
        }

    def process_file_to_chunks(self, file_path: str) -> List[Dict]:
        chunks = self.doc_processor.process_file(file_path)
        return self.preprocessor.preprocess_chunks(chunks)

    def create_workspace_document(self, user_id: str, document_id: Optional[str] = None) -> str:
        if document_id is None:
            document_id = str(uuid.uuid4())

        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        if not chunks_file.exists():
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)

        self._set_owner(document_id, user_id)
        return document_id

    def _load_document_chunks(self, document_id: str) -> Optional[List[Dict]]:
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        if not chunks_file.exists():
            return None
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_document_chunks(self, document_id: str, chunks: List[Dict]):
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)

    def append_chunks(self, document_id: str, user_id: str, new_chunks: List[Dict]) -> Optional[int]:
        if not self._is_owned_by_user(document_id, user_id):
            return None

        existing_chunks = self._load_document_chunks(document_id)
        if existing_chunks is None:
            return None

        next_chunk_id = len(existing_chunks)
        normalized_chunks: List[Dict] = []
        for chunk in new_chunks:
            normalized = chunk.copy()
            normalized["chunk_id"] = next_chunk_id
            next_chunk_id += 1
            normalized["text"] = self.preprocessor.clean_text(normalized.get("text", ""))
            topic = normalized.get("topic", "Content")
            normalized["combined_text"] = f"{topic}: {normalized['text']}"
            normalized_chunks.append(normalized)

        merged = existing_chunks + normalized_chunks
        self._save_document_chunks(document_id, merged)
        return len(merged)

    def chunk_text_content(self, text: str, source_name: str, topic: str = "Transcript", chunk_size: int = 500, overlap: int = 120) -> List[Dict]:
        text = self.preprocessor.clean_text(text)
        if not text:
            return []

        segments = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[Dict] = []
        current = ""

        for segment in segments:
            part = segment.strip()
            if not part:
                continue

            if len(current) + len(part) + 1 > chunk_size and current:
                chunks.append({
                    "doc_id": source_name,
                    "source_name": source_name,
                    "page": "N/A",
                    "topic": topic,
                    "chunk_id": len(chunks),
                    "text": current.strip(),
                })
                tail = current[-overlap:] if len(current) > overlap else current
                current = f"{tail} {part}".strip()
            else:
                current = f"{current} {part}".strip()

        if current:
            chunks.append({
                "doc_id": source_name,
                "source_name": source_name,
                "page": "N/A",
                "topic": topic,
                "chunk_id": len(chunks),
                "text": current.strip(),
            })

        return self.preprocessor.preprocess_chunks(chunks)
    
    def get_document(self, document_id: str, user_id: str) -> Optional[Dict]:
        if not self._is_owned_by_user(document_id, user_id):
            return None
            
        chunks = self._load_document_chunks(document_id)
        if chunks is None:
            return None
        
        return {
            "document_id": document_id,
            "filename": self._strip_uuid_prefix(chunks[0]["source_name"]) if chunks else "unknown",
            "num_chunks": len(chunks),
            "chunks": chunks
        }
    
    @staticmethod
    def _strip_uuid_prefix(name: str) -> str:
        """Remove the leading {uuid}_ prefix from source_name."""
        parts = name.split("_", 1)
        if len(parts) == 2:
            try:
                uuid.UUID(parts[0])
                return parts[1]
            except ValueError:
                pass
        return name
    
    def get_chunks(self, document_id: str, user_id: str, chunk_ids: Optional[List[int]] = None) -> Optional[List[Dict]]:
       
        document = self.get_document(document_id, user_id)
        
        if document is None:
            return None
        
        chunks = document["chunks"]
        
        if chunk_ids is None:
            return chunks
        
        filtered_chunks = [chunk for chunk in chunks if chunk["chunk_id"] in chunk_ids]
        return filtered_chunks
    
    def delete_document(self, document_id: str, user_id: str) -> bool:
        registry = self._load_registry()
        if self._extract_owner_id(registry.get(document_id)) != user_id:
            return False

        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        
        if chunks_file.exists():
            chunks_file.unlink()
            if document_id in registry:
                del registry[document_id]
                self._save_registry(registry)
            return True
        
        return False
    
    def list_documents(self, user_id: str) -> List[Dict]:

        documents = []
        registry = self._load_registry()
        
        for chunks_file in self.data_dir.glob("*_chunks.json"):
            document_id = chunks_file.stem.replace("_chunks", "")
            
            if self._extract_owner_id(registry.get(document_id)) != user_id:
                continue
                
            document = self.get_document(document_id, user_id)
            
            if document:
                documents.append({
                    "document_id": document_id,
                    "filename": document["filename"],
                    "num_chunks": document["num_chunks"]
                })
        
        return documents
