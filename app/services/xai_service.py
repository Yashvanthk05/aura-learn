import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import re


class ExplainableExtractiveService:

    def __init__(self, extractive_summarizer):
        self.summarizer = extractive_summarizer
        self.device = extractive_summarizer.device
        self.encoder = extractive_summarizer.encoder
        self.model = extractive_summarizer.model

    def explain_extractive(
        self,
        text: str,
        num_sentences: int = 3,
        generate_lrp: bool = False,
    ) -> Dict:
        """
        Full explanation pipeline for extractive summarization.
        Returns sentence scores, attention weights, and sensitivity analysis.
        """
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]

        if not sentences:
            return {"error": "No valid sentences found in text"}

        if len(sentences) <= num_sentences:
            all_indices = list(range(len(sentences)))
            sentence_explanations = [
                {
                    "index": i,
                    "text": s,
                    "importance_score": 1.0,
                    "is_selected": True,
                    "sensitivity": 0.0,
                    "word_count": len(s.split()),
                }
                for i, s in enumerate(sentences)
            ]

            lrp_explanation = None
            if generate_lrp:
                lrp_explanation = self._generate_lrp_explanation(sentences, all_indices)

            explanation_methods = ["all_sentences_selected"]
            if lrp_explanation:
                explanation_methods.append("layer_wise_relevance_propagation (Gradient x Input)")

            return {
                "summary": " ".join(sentences),
                "num_sentences_input": len(sentences),
                "num_sentences_selected": len(sentences),
                "selected_indices": all_indices,
                "average_score_selected": 1.0,
                "average_score_all": 1.0,
                "score_distribution": {"min": 1.0, "max": 1.0, "std": 0.0},
                "sentences": sentence_explanations,
                "lrp_explanation": lrp_explanation,
                "explanation_methods": explanation_methods,
                "xai_type": "post-hoc + deep_explanation",
            }

        # 1. Get base scores
        base_scores = self._get_scores(sentences)

        # 2. Get attention weights
        attention_weights = self._get_attention_weights(sentences)

        # 3. Sensitivity analysis (leave-one-out)
        sensitivity = self._sensitivity_analysis(sentences, base_scores)

        # 4. Select top sentences
        top_indices = sorted(
            np.argsort(base_scores)[-num_sentences:][::-1]
        )
        summary = " ".join([sentences[i] for i in sorted(top_indices)])

        # Build per-sentence explanation
        sentence_explanations = []
        for i, sent in enumerate(sentences):
            explanation = {
                "index": i,
                "text": sent,
                "importance_score": round(float(base_scores[i]), 4),
                "is_selected": i in top_indices,
                "sensitivity": round(float(sensitivity[i]), 4),
                "word_count": len(sent.split()),
            }

            # Add attention info: which other sentences this one attends to most
            if attention_weights is not None and i < len(attention_weights):
                attn_row = attention_weights[i][:len(sentences)]
                top_attended = int(np.argmax(attn_row))
                explanation["most_attended_sentence"] = top_attended
                explanation["attention_to_others"] = [
                    round(float(a), 4) for a in attn_row
                ]

            sentence_explanations.append(explanation)

        # Summary-level explanation
        avg_selected_score = float(np.mean([base_scores[i] for i in top_indices]))
        avg_all_score = float(np.mean(base_scores))

        # 5. Generate LRP explanation if requested
        lrp_explanation = None
        if generate_lrp:
            lrp_explanation = self._generate_lrp_explanation(sentences, top_indices)

        explanation_methods = [
            "importance_scoring",
            "attention_weights",
            "sensitivity_analysis",
        ]
        if lrp_explanation:
            explanation_methods.append("layer_wise_relevance_propagation (Gradient x Input)")

        return {
            "summary": summary,
            "num_sentences_input": len(sentences),
            "num_sentences_selected": len(top_indices),
            "selected_indices": [int(i) for i in top_indices],
            "average_score_selected": round(avg_selected_score, 4),
            "average_score_all": round(avg_all_score, 4),
            "score_distribution": {
                "min": round(float(np.min(base_scores)), 4),
                "max": round(float(np.max(base_scores)), 4),
                "std": round(float(np.std(base_scores)), 4),
            },
            "sentences": sentence_explanations,
            "lrp_explanation": lrp_explanation,
            "explanation_methods": explanation_methods,
            "xai_type": "post-hoc + deep_explanation",
        }

    def _get_scores(self, sentences: List[str]) -> np.ndarray:
        """Get importance scores for each sentence from the model."""
        with torch.no_grad():
            embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
            embeddings_t = (
                torch.tensor(embeddings, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            lengths = torch.tensor([len(sentences)])
            mask = torch.ones(1, len(sentences), dtype=torch.bool).to(self.device)

            scores = self.model(embeddings_t, lengths, mask)
            return scores.squeeze(0).cpu().numpy()

    def _generate_lrp_explanation(
        self, sentences: List[str], selected_indices: List[int]
    ) -> Optional[Dict]:
        """
        Calculates feature attribution mathematically aligned to Layer-wise Relevance 
        Propagation (Gradient * Input) using the captum library to show how parts of 
        the document contributed to the summarizer's selection of a specific sentence.
        """
        try:
            import captum.attr as attr
            
            # Get embeddings
            embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
            embeddings_t = (
                torch.tensor(embeddings, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            embeddings_t.requires_grad_(True)
            
            lengths = torch.tensor([len(sentences)])
            mask = torch.ones(1, len(sentences), dtype=torch.bool).to(self.device)

            # Wrapper for captum
            def forward_func(emb):
                return self.model(emb, lengths, mask)

            # Temporarily set to train mode for CuDNN RNN backward compatibility, 
            # but we won't actually step the optimizer.
            was_training = self.model.training
            self.model.train()

            # Array to hold attributions. Shape: [num_selected, num_input]
            attributions_matrix = []
            
            input_x_grad = attr.InputXGradient(forward_func)
            
            for idx in selected_indices:
                # Captum natively handles the backward pass based on the target class (idx)
                # It expects a standard python int for target
                self.model.zero_grad()
                grad_x_input = input_x_grad.attribute(embeddings_t, target=int(idx))

                # Sum across the embedding dimension to get relevance per sentence
                sentence_relevance = grad_x_input.sum(dim=-1).squeeze(0).cpu().detach().numpy()
                
                # Optional: normalize to make the values visually proportional 
                # to their absolute percentage influence
                total_abs_relevance = np.sum(np.abs(sentence_relevance))
                if total_abs_relevance > 0:
                    sentence_relevance = sentence_relevance / total_abs_relevance
                    
                attributions_matrix.append([round(float(val), 4) for val in sentence_relevance])

            # Restore original mode
            if not was_training:
                self.model.eval()

            return {
                "selected_sentences": selected_indices,
                "input_sentences": len(sentences),
                "feature_attributions": attributions_matrix
            }
            
        except ImportError:
            print("WARN: Captum is not installed. LRP explanation skipped.")
            return None
        except Exception as e:
            print(f"WARN: LRP extraction failed: {e}")
            return None

    def _get_attention_weights(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Extract attention weights from the MultiheadAttention layer."""
        try:
            with torch.no_grad():
                embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
                embeddings_t = (
                    torch.tensor(embeddings, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

                batch_size, seq_len, _ = embeddings_t.shape
                pos = (
                    torch.arange(seq_len, device=self.device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
                x = embeddings_t + self.model.pos_emb(pos)

                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

                lengths_cpu = torch.tensor([len(sentences)])
                packed_x = pack_padded_sequence(
                    x, lengths_cpu.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.model.lstm(packed_x)
                lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
                lstm_out = self.model.layer_norm1(lstm_out)

                mask = torch.ones(1, len(sentences), dtype=torch.bool).to(self.device)
                _, attn_weights = self.model.attention(
                    lstm_out,
                    lstm_out,
                    lstm_out,
                    key_padding_mask=~mask,
                    need_weights=True,
                    average_attn_weights=True,
                )

                return attn_weights.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"WARN: Attention extraction failed: {e}")
            return None

    def _sensitivity_analysis(
        self, sentences: List[str], base_scores: np.ndarray
    ) -> np.ndarray:
        """
        Leave-one-out sensitivity analysis.
        Measures how much removing each sentence affects the scores of others.
        Higher value = removing this sentence causes larger score changes.
        """
        sensitivities = np.zeros(len(sentences))

        for i in range(len(sentences)):
            reduced = [s for j, s in enumerate(sentences) if j != i]
            if len(reduced) < 2:
                sensitivities[i] = float(base_scores[i])
                continue

            reduced_scores = self._get_scores(reduced)

            # Compare base scores (excluding i) with reduced scores
            base_without_i = np.delete(base_scores, i)
            score_diff = np.abs(base_without_i - reduced_scores)
            sensitivities[i] = float(np.mean(score_diff))

        # Normalize to [0, 1]
        max_val = sensitivities.max()
        if max_val > 0:
            sensitivities = sensitivities / max_val

        return sensitivities


class ExplainableSearchService:
    """
    Provides explanations for hybrid search results.
    Breaks down scoring contributions from FAISS, BM25, and TF-IDF.
    """

    def explain_search(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5,
    ) -> Dict:
        """Explain why each search result was returned and ranked."""
        explained_results = []

        for rank, result in enumerate(results[:top_k]):
            breakdown = result.get("score_breakdown", {})
            total = breakdown.get("combined", result.get("score", 0.0))

            # Determine dominant scoring method
            faiss_score = breakdown.get("faiss", 0.0)
            bm25_score = breakdown.get("bm25", 0.0)
            tfidf_score = breakdown.get("tfidf", 0.0)

            scores_map = {
                "semantic_similarity (FAISS)": faiss_score,
                "keyword_match (BM25)": bm25_score,
                "term_frequency (TF-IDF)": tfidf_score,
            }
            dominant = max(scores_map, key=scores_map.get)

            # Word overlap analysis
            query_words = set(query.lower().split())
            text_words = set(result.get("text", "").lower().split())
            overlap = query_words & text_words
            overlap_ratio = (
                len(overlap) / len(query_words) if query_words else 0.0
            )

            explained_results.append({
                "rank": rank + 1,
                "chunk_id": result.get("chunk_id"),
                "topic": result.get("topic", "Unknown"),
                "page": result.get("page", "N/A"),
                "text_preview": (
                    result["text"][:200] + "..."
                    if len(result.get("text", "")) > 200
                    else result.get("text", "")
                ),
                "combined_score": round(total, 4),
                "score_breakdown": {
                    "faiss_semantic": round(faiss_score, 4),
                    "bm25_keyword": round(bm25_score, 4),
                    "tfidf_term": round(tfidf_score, 4),
                },
                "dominant_scoring_method": dominant,
                "query_word_overlap": list(overlap),
                "word_overlap_ratio": round(overlap_ratio, 4),
                "explanation": _generate_search_explanation(
                    dominant, faiss_score, bm25_score, tfidf_score, overlap_ratio
                ),
            })

        return {
            "query": query,
            "total_results_explained": len(explained_results),
            "results": explained_results,
            "scoring_weights": {
                "faiss (semantic)": "50%",
                "bm25 (keyword)": "30%",
                "tfidf (term frequency)": "20%",
            },
            "xai_type": "transparent_approximation",
        }


class ExplainableAbstractiveService:
    """
    Provides token-level attribution for T5 abstractive summarizer
    using input-reduction sensitivity analysis.
    """

    def __init__(self, abstractive_summarizer):
        self.summarizer = abstractive_summarizer
        self.tokenizer = abstractive_summarizer.tokenizer
        self.model = abstractive_summarizer.model
        self.device = abstractive_summarizer.device

    def explain_abstractive(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        generate_shap: bool = False,
    ) -> Dict:
        """
        Explain abstractive summary by measuring which input sentences
        contribute most to the generated output. Also computes SHAP values if requested.
        """
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]

        if not sentences:
            return {"error": "No valid sentences found"}

        # Generate full summary
        full_summary = self.summarizer.summarize(
            text, max_length=max_length, min_length=min_length
        )

        # Measure each sentence's contribution via leave-one-out
        contributions = []
        for i in range(len(sentences)):
            reduced_text = " ".join(s for j, s in enumerate(sentences) if j != i)
            if not reduced_text.strip():
                contributions.append({
                    "index": i,
                    "text": sentences[i],
                    "contribution_score": 1.0,
                    "without_summary": "",
                })
                continue

            reduced_summary = self.summarizer.summarize(
                reduced_text, max_length=max_length, min_length=min_length
            )

            # Measure how much the summary changed
            change_score = _text_dissimilarity(full_summary, reduced_summary)

            contributions.append({
                "index": i,
                "text": sentences[i],
                "contribution_score": round(change_score, 4),
                "without_summary": reduced_summary,
            })

        # Normalize contribution scores
        max_contrib = max(c["contribution_score"] for c in contributions) if contributions else 1.0
        if max_contrib > 0:
            for c in contributions:
                c["normalized_contribution"] = round(
                    c["contribution_score"] / max_contrib, 4
                )
        else:
            for c in contributions:
                c["normalized_contribution"] = 0.0

        # Sort by contribution
        contributions.sort(key=lambda x: x["contribution_score"], reverse=True)

        # Get decoder token-level log probabilities
        token_probs = self._get_token_probabilities(text, full_summary)

        # Generate SHAP text attribution if requested
        shap_explanation = None
        if generate_shap:
            shap_explanation = self._generate_shap_explanation(text)

        explanation_methods = [
            "leave_one_out_attribution",
            "token_confidence_scores",
        ]
        if shap_explanation:
            explanation_methods.append("shap_text_attribution")

        return {
            "original_text_sentences": len(sentences),
            "summary": full_summary,
            "summary_word_count": len(full_summary.split()),
            "compression_ratio": round(
                len(full_summary.split()) / max(len(text.split()), 1), 4
            ),
            "sentence_contributions": contributions,
            "most_influential_sentence": contributions[0] if contributions else None,
            "token_confidence": token_probs,
            "shap_explanation": shap_explanation,
            "explanation_methods": explanation_methods,
            "xai_type": "post-hoc_sensitivity_analysis",
        }

    def _generate_shap_explanation(self, text: str) -> Optional[Dict]:
        """Generate SHAP values for the summarization using huggingface pipeline."""
        try:
            import shap
            from transformers import pipeline
            from packaging import version
            
            # Create a pipeline from the existing model and tokenizer
            pipe = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=self.device.index if self.device.type == "cuda" else -1
            )
            
            explainer = shap.Explainer(pipe)
            # Use a smaller text if it's too long to prevent extremely slow computation
            # SHAP is O(N^2) or worse for text generation
            max_words_shap = 150
            words = text.split()
            if len(words) > max_words_shap:
                short_text = " ".join(words[:max_words_shap])
            else:
                short_text = text
                
            shap_values = explainer([short_text])
            
            # Format outputs
            return {
                "input_tokens": list(shap_values.data[0]),
                "output_tokens": list(shap_values.output_names),
                "shap_values": [list(val) for val in shap_values.values[0]]
            }
        except Exception as e:
            print(f"WARN: SHAP explanation generation failed: {e}")
            return None

    def _get_token_probabilities(
        self, input_text: str, summary: str
    ) -> List[Dict]:
        """Get per-token generation confidence from the decoder."""
        try:
            input_ids = self.tokenizer(
                "summarize: " + input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).input_ids.to(self.device)

            target_ids = self.tokenizer(
                summary, return_tensors="pt", truncation=True, max_length=512
            ).input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, labels=target_ids
                )
                logits = outputs.logits  # (1, seq_len, vocab_size)
                probs = torch.softmax(logits, dim=-1)

            token_results = []
            for idx in range(target_ids.shape[1]):
                token_id = target_ids[0, idx].item()
                token_text = self.tokenizer.decode([token_id])
                confidence = float(probs[0, idx, token_id].cpu())

                # Skip special tokens
                if token_id in self.tokenizer.all_special_ids:
                    continue

                token_results.append({
                    "token": token_text.strip(),
                    "confidence": round(confidence, 4),
                    "is_high_confidence": confidence > 0.5,
                })

            return token_results
        except Exception as e:
            print(f"WARN: Token probability extraction failed: {e}")
            return []


def _text_dissimilarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard dissimilarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 0.0
    union = words_a | words_b
    intersection = words_a & words_b
    if not union:
        return 0.0
    return 1.0 - (len(intersection) / len(union))


def _generate_search_explanation(
    dominant: str,
    faiss: float,
    bm25: float,
    tfidf: float,
    overlap_ratio: float,
) -> str:
    """Generate a human-readable explanation for a search result."""
    parts = []

    if "FAISS" in dominant:
        parts.append(
            "This result was primarily matched based on semantic meaning similarity."
        )
    elif "BM25" in dominant:
        parts.append(
            "This result was primarily matched based on keyword overlap."
        )
    elif "TF-IDF" in dominant:
        parts.append(
            "This result was primarily matched based on term frequency analysis."
        )

    if overlap_ratio > 0.5:
        parts.append(
            f"High direct word overlap ({overlap_ratio:.0%}) with the query."
        )
    elif overlap_ratio > 0.2:
        parts.append(
            f"Moderate word overlap ({overlap_ratio:.0%}) with the query."
        )
    else:
        parts.append(
            "Low direct word overlap; match is primarily semantic."
        )

    total = faiss + bm25 + tfidf
    if total > 0:
        parts.append(
            f"Score composition: semantic {faiss/total:.0%}, "
            f"keyword {bm25/total:.0%}, term-freq {tfidf/total:.0%}."
        )

    return " ".join(parts)