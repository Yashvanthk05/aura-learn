import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ExtractiveModel(torch.nn.Module):

    def __init__(self, input_dim=384, hidden_dim=256, num_layers=2, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.pos_emb = torch.nn.Embedding(100, input_dim)
        
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = torch.nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_dim * 2)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x, lengths, mask):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        batch_size, seq_len, _ = x.shape
        
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_emb(pos)
        
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        lstm_out = self.layer_norm1(lstm_out)
        
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=~mask,
            need_weights=False
        )
        
        combined = self.layer_norm2(lstm_out + attn_out)
        scores = self.classifier(combined).squeeze(-1)
        
        return scores

class ExtractiveSummarizer:

    def __init__(self, model_path: str, encoder_name: str = 'all-MiniLM-L6-v2'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SentenceTransformer(encoder_name)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = checkpoint['config']
        
        self.model = ExtractiveModel(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Extractive model loaded on {self.device}")
        # if 'rouge_scores' in checkpoint:
        #     print(f"Model ROUGE-1 score: {checkpoint['rouge_scores']['rouge1']:.4f}")
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        min_confidence: float = 0.0,
        diverse_selection: bool = True,
        diversity_penalty: float = 0.3,
        preserve_order: bool = True
    ) -> str:
        
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        with torch.no_grad():
            embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
            embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            lengths = torch.tensor([len(sentences)])
            mask = torch.ones(1, len(sentences), dtype=torch.bool).to(self.device)
            
            scores = self.model(embeddings, lengths, mask)
            scores = scores.squeeze(0).cpu().numpy()
        
        if diverse_selection:
            selected = self._diverse_selection(
                scores, sentences, num_sentences, min_confidence, diversity_penalty
            )
        else:
            selected = self._simple_selection(scores, sentences, num_sentences, min_confidence)
        
        if not selected:
            selected = [np.argmax(scores)]
        
        if preserve_order:
            selected = sorted(selected)
        
        summary_sentences = [sentences[i] for i in selected]
        return " ".join(summary_sentences)
    
    def summarize_with_scores(self, text: str, num_sentences: int = 3) -> Tuple[str, Dict]:
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        
        with torch.no_grad():
            embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
            embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            lengths = torch.tensor([len(sentences)])
            mask = torch.ones(1, len(sentences), dtype=torch.bool).to(self.device)
            
            scores = self.model(embeddings, lengths, mask)
            scores = scores.squeeze(0).cpu().numpy()
        
        top_indices = np.argsort(scores)[-num_sentences:][::-1]
        selected_indices = sorted(top_indices)
        
        scores_dict = {
            i: {
                'score': float(scores[i]),
                'sentence': sentences[i]
            }
            for i in selected_indices
        }
        
        summary_sentences = [sentences[i] for i in selected_indices]
        return " ".join(summary_sentences), scores_dict
    
    @staticmethod
    def _simple_selection(scores, sentences, num_sentences, min_confidence):
        top_indices = np.argsort(scores)[-num_sentences:][::-1]
        return [idx for idx in top_indices if scores[idx] >= min_confidence]
    
    @staticmethod
    def _diverse_selection(scores, sentences, num_sentences, min_confidence, diversity_penalty):
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = encoder.encode(sentences)
        
        selected = []
        available_scores = scores.copy()
        
        for _ in range(num_sentences):
            if len(available_scores) == 0:
                break
            
            best_idx = np.argmax(available_scores)
            
            if available_scores[best_idx] < min_confidence and len(selected) > 0:
                break
            
            selected.append(int(best_idx))
            available_scores[best_idx] = -float('inf')
    
            for idx in range(len(available_scores)):
                if idx not in selected:
                    similarity = np.dot(embeddings[best_idx], embeddings[idx])
                    similarity /= (np.linalg.norm(embeddings[best_idx]) * np.linalg.norm(embeddings[idx]))
                    available_scores[idx] -= diversity_penalty * similarity
        
        return selected

    def get_sentence_scores(self, sentences: List[str]) -> np.ndarray:

        if not sentences:
            return np.array([])
        
        embeddings = self.encoder.encode(sentences)
        embeddings_tensor = torch.FloatTensor(embeddings).unsqueeze(0).to(self.device)
        
        lengths = torch.tensor([len(sentences)], dtype=torch.long)
        mask = torch.ones((1, len(sentences)), dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            scores = self.model(embeddings_tensor, lengths, mask)
        
        return scores[0].cpu().numpy()
