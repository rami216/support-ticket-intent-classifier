from transformers import DistilBertForSequenceClassification

def build_distilbert_model(
  model_name:str,
  num_classes:int
):
  model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes
  )
  return model