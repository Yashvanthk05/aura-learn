import fitz
import re
import os
import json
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict

class PDFProcessor:

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        body_font_size = self._analyze_body_font_size(doc)
        
        chunks = self._get_structure_based_chunks(doc, pdf_path, body_font_size)
        
        if len(chunks)<3:
            chunks = self._get_paragraph_chunks(doc, pdf_path)
        
        doc.close()
        return chunks
    
    def _analyze_body_font_size(self, doc) -> float:
        font_counts = {}
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            size = round(span["size"], 1)
                            text = span["text"].strip()
                            if len(text) > 1:
                                font_counts[size] = font_counts.get(size, 0) + len(text)
        
        if not font_counts:
            return 11.0
        return max(font_counts, key=font_counts.get)
    
    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def _generate_caption(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return f"[image: {caption}]"
        except Exception as e:
            return ""
        
    def _get_structure_based_chunks(self, doc, pdf_path: str, body_font_size: float) -> List[Dict]:
        chunks = []
        current_topic = "General Content"
        current_text = []
        current_topic_page = 1
        
        header_threshold = body_font_size + 1.2
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            blocks.sort(key=lambda b: b["bbox"][1])
            
            for b in blocks:
                if b["type"] == 1:
                    caption = self._generate_caption(b["image"])
                    if caption:
                        current_text.append(caption)
                    continue
                
                if "lines" in b:
                    for line in b["lines"]:
                        line_text = "".join([s["text"] for s in line["spans"]])
                        clean_line = self._clean_text(line_text)
                        if not clean_line:
                            continue
                        
                        line_max_size = max([s["size"] for s in line["spans"]])
                        
                        is_header = (
                            (line_max_size > header_threshold) and 
                            (len(clean_line.split()) < 10) and 
                            (not clean_line.strip().startswith(('"', """, """, "'"))) and
                            (clean_line[0].isupper() or clean_line[0].isdigit())
                        )
                        
                        if is_header:
                            if current_text:
                                chunks.append({
                                    "doc_id": os.path.basename(pdf_path),
                                    "source_name": os.path.basename(pdf_path),
                                    "page": current_topic_page,
                                    "topic": current_topic,
                                    "chunk_id": len(chunks),
                                    "text": " ".join(current_text)
                                })
                                current_text = []
                            
                            current_topic = clean_line
                            current_topic_page = page_num + 1
                        else:
                            current_text.append(clean_line)
        
        if current_text:
            chunks.append({
                "doc_id": os.path.basename(pdf_path),
                "source_name": os.path.basename(pdf_path),
                "page": current_topic_page,
                "topic": current_topic,
                "chunk_id": len(chunks),
                "text": " ".join(current_text)
            })
        
        return chunks
    
    def _get_paragraph_chunks(self, doc, pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n"
        
        paragraphs = full_text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            clean_para = self._clean_text(para)
            if not clean_para:
                continue
            
            if len(current_chunk) + len(clean_para) > chunk_size:
                chunks.append({
                    "doc_id": os.path.basename(pdf_path),
                    "source_name": os.path.basename(pdf_path),
                    "page": "N/A",
                    "topic": "Paragraph Chunk",
                    "chunk_id": len(chunks),
                    "text": current_chunk.strip()
                })
                
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + clean_para
            else:
                current_chunk += " " + clean_para
        
        if current_chunk:
            chunks.append({
                "doc_id": os.path.basename(pdf_path),
                "source_name": os.path.basename(pdf_path),
                "page": "N/A",
                "topic": "Paragraph Chunk",
                "chunk_id": len(chunks),
                "text": current_chunk.strip()
            })
        
        return chunks
    

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def preprocess_chunks(chunks: List[Dict]) -> List[Dict]:
        processed_chunks = []
        for chunk in chunks:
            new_chunk = chunk.copy()
            new_chunk["text"] = TextPreprocessor.clean_text(chunk["text"])
            new_chunk["combined_text"] = f"{chunk['topic']}: {chunk['text']}"
            processed_chunks.append(new_chunk)
        return processed_chunks
