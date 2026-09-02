import torch
import torch.nn as nn

from src.models.rnn import RNNClassifier

def main()->None:
  model = RNNClassifier(
    vocab_size=20,
    embedding_dim=5,
    hidden_size=6,
    num_classes=7,
    padding_idx=0
  )
  input_ids = torch.tensor(
    [
      [2,3,4,0],
      [5,6,7,8]
    ],
    dtype=torch.long
  )
  lengths = torch.tensor(
    [3,4],
    dtype=torch.long
  )
  labels = torch.tensor(
    [4,0],
    dtype=torch.long
  )
  
  criterion = nn.CrossEntropyLoss()
  
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
  )
  
  optimizer.zero_grad()
  
  logits = model(
    input_ids=input_ids,
    lengths=lengths
  )
  
  loss = criterion(
    logits,labels
  )
  loss.backward()
  
  optimizer.step()
  
  print("Loss:")
  print(loss.item())
  
  
if __name__=="__main__":
  main()