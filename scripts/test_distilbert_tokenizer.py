from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"


def main()->None:
  tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
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
  
  tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
  )
  
  print("Text:")
  print(text)

  print("\nTokens:")
  print(tokens)

  print("\nInput IDs:")
  print(input_ids)

  print("\nAttention Mask:")
  print(attention_mask)
  print("\nShapes:")
  print("input_ids:", input_ids.shape)
  print("attention_mask:", attention_mask.shape)



if __name__=="__main__":
  main()