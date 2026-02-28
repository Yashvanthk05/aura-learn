import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Optional

class AbstractiveSummarizer:
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading T5 model from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Abstractive model loaded on {self.device}")
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        num_beams: int = 8,
        length_penalty: float = 2.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        diversity_penalty: float = 1.5,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 3.0
    ) -> str:
        
        input_text = "summarize: " + text
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                num_beam_groups=4,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                diversity_penalty=diversity_penalty,
                do_sample=False
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(self, texts: list, max_length: int = 150, min_length: int = 40,**kwargs) -> list:
        summaries = []
        for text in texts:
            summary = self.summarize(text, max_length, min_length, **kwargs)
            summaries.append(summary)
        return summaries
