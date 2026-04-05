"""
Document management service.
"""
import json
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
                return json.load(f)
        return {}
        
    def _save_registry(self, registry: Dict):
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def process_document(self, pdf_path: str, user_id: str, document_id: Optional[str] = None) -> Dict:

        if document_id is None:
            document_id = str(uuid.uuid4())
        
        chunks = self.doc_processor.process_file(pdf_path)
        
        processed_chunks = self.preprocessor.preprocess_chunks(chunks)
        
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2)
            
        registry = self._load_registry()
        registry[document_id] = user_id
        self._save_registry(registry)
        
        return {
            "document_id": document_id,
            "filename": Path(pdf_path).name,
            "num_chunks": len(processed_chunks),
            "chunks": processed_chunks
        }
    
    def get_document(self, document_id: str, user_id: str) -> Optional[Dict]:
        registry = self._load_registry()
        if registry.get(document_id) != user_id:
            return None
            
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        
        if not chunks_file.exists():
            return None
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
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
        if registry.get(document_id) != user_id:
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
            
            if registry.get(document_id) != user_id:
                continue
                
            document = self.get_document(document_id, user_id)
            
            if document:
                documents.append({
                    "document_id": document_id,
                    "filename": document["filename"],
                    "num_chunks": document["num_chunks"]
                })
        
        return documents
