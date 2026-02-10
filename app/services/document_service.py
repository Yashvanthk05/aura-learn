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
    
    def process_document(self, pdf_path: str, document_id: Optional[str] = None) -> Dict:

        if document_id is None:
            document_id = str(uuid.uuid4())
        
        chunks = self.doc_processor.process_file(pdf_path)
        
        processed_chunks = self.preprocessor.preprocess_chunks(chunks)
        
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2)
        
        return {
            "document_id": document_id,
            "filename": Path(pdf_path).name,
            "num_chunks": len(processed_chunks),
            "chunks": processed_chunks
        }
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        
        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        
        if not chunks_file.exists():
            return None
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        return {
            "document_id": document_id,
            "filename": chunks[0]["source_name"] if chunks else "unknown",
            "num_chunks": len(chunks),
            "chunks": chunks
        }
    
    def get_chunks(self, document_id: str, chunk_ids: Optional[List[int]] = None) -> Optional[List[Dict]]:
       
        document = self.get_document(document_id)
        
        if document is None:
            return None
        
        chunks = document["chunks"]
        
        if chunk_ids is None:
            return chunks
        
        filtered_chunks = [chunk for chunk in chunks if chunk["chunk_id"] in chunk_ids]
        return filtered_chunks
    
    def delete_document(self, document_id: str) -> bool:

        chunks_file = self.data_dir / f"{document_id}_chunks.json"
        
        if chunks_file.exists():
            chunks_file.unlink()
            return True
        
        return False
    
    def list_documents(self) -> List[Dict]:

        documents = []
        
        for chunks_file in self.data_dir.glob("*_chunks.json"):
            document_id = chunks_file.stem.replace("_chunks", "")
            document = self.get_document(document_id)
            
            if document:
                documents.append({
                    "document_id": document_id,
                    "filename": document["filename"],
                    "num_chunks": document["num_chunks"]
                })
        
        return documents
