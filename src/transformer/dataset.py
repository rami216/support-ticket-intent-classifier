import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class DistilBertTicketDataset(Dataset):
  def __init__(
    self,
    texts:list[str],
    labels:list[str],
    tokenizer:PreTrainedTokenizerBase,
    label_to_id:dict[str,int],
    max_length:int=128
  ):
    self.texts=texts
    self.labels=labels
    self.tokenizer=tokenizer
    self.label_to_id = label_to_id
    self.max_length=max_length
    
  def __len__(self)->int:
    return len(self.texts)
  
  def __getitem__(self,index:int)->dict[str,torch.Tensor]:
    text = self.texts[index]
    label = self.labels[index]
    
    encoded = self.tokenizer(
      text,
      padding="max_length",
      truncation=True,
      max_length=self.max_length,
      return_tensors = "pt"
    )
    input_ids = encoded["input_ids"].squeeze(0)
    
    attenton_mask = encoded["attention_mask"].squeeze(0)
    
    label_id = self.label_to_id[label]
    
    return {
      "input_ids":input_ids,
      "attention_mask":attenton_mask,
      "label":torch.tensor(
        label_id,
        dtype=torch.long
      )
    }