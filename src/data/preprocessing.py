import re

def normalize_text(text:str)->str:
  text = text.lower()
  text = text.strip()
  
  return text


def tokenize(text:str)->list[str]:
  text = normalize_text(text)
  
  tokens = re.findall(r"\b\w+\b",text)
  
  return tokens


