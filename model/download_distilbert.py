import os
from transformers import DistilBertModel, DistilBertTokenizer




model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained('./')
model.save_pretrained('./')

print("Model and tokenizer saved successfully.")
