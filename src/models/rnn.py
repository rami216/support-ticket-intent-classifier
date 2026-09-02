import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
  def __init__(
    self,
    vocab_size:int,
    embedding_dim:int,
    hidden_size:int,
    num_classes:int,
    padding_idx:int,
  ):
    super().__init__()
    
    self.embeddings = nn.Embedding(
      num_embeddings = vocab_size,
      embedding_dim=embedding_dim,
      padding_idx=padding_idx
    )
    
    self.input_to_hidden = nn.Linear(
      embedding_dim,hidden_size
    )
    
    self.hidden_to_hidden = nn.Linear(
      hidden_size,
      hidden_size,
      bias=False,
    )
    self.classifier = nn.Linear(
      hidden_size,
      num_classes
    )
    
  def forward(
    self,
    input_ids:torch.Tensor,
    lengths:torch.Tensor,
  )->torch.Tensor:
    
    embeddings = self.embeddings(input_ids)
    
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    
    hidden = torch.zeros(
      batch_size,
      self.hidden_to_hidden.in_features,
      device = input_ids.device
    )
    for t in range(sequence_length):
      x_t = embeddings[:,t,:]
      
      new_hidden = torch.tanh(
        self.input_to_hidden(x_t)
        +
        self.hidden_to_hidden(hidden)
      )
      active = (t<lengths).unsqueeze(1)
      hidden = torch.where(
        active,
        new_hidden,
        hidden
      )
    
    logits = self.classifier(hidden)
    return logits
    
  