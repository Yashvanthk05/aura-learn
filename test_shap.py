import shap
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
model_id = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)
pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
explainer = shap.Explainer(pipe)
text = "The quick brown fox jumps over the lazy dog. The dog is very lazy."
shap_values = explainer([text])
print("data:", shap_values.data[0])
print("output_names:", shap_values.output_names)
print("values shape:", shap_values.values[0].shape)
print("values:", shap_values.values[0])