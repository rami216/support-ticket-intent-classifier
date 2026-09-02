import torch
from transformers import AutoTokenizer

from src.models.distilbert import build_distilbert_model


MODEL_NAME = "distilbert-base-uncased"
NUM_CLASSES = 27

def main()->None:
  tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
  )
  model = build_distilbert_model(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES
  )
  text = "I forgot my password"
  
  encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=16,
    return_tensors="pt"
  )
  input_ids = encoded["input_ids"]
  attention_mask = encoded["attention_mask"]
  
  
  with torch.no_grad():
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    

  logits = outputs.logits
  print("input_ids shape:")
  print(input_ids.shape)

  print("\nattention_mask shape:")
  print(attention_mask.shape)

  print("\nlogits shape:")
  print(logits.shape)

  print("\nlogits:")
  print(logits)
  
  
if __name__ == "__main__":
    main()