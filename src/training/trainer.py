import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.metrics import calculate_metrics


def train_one_epoch(
  model:nn.Module,
  train_loader:DataLoader,
  criterion:nn.Module,
  optimizer:torch.optim.Optimizer,
  device:torch.device,
)->float:
  
  model.train()
  
  total_loss = 0.0
  total_correct=0
  total_examples = 0
  
  for batch in train_loader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    lengths = batch["lengths"].to(device)
    
    optimizer.zero_grad()
    
    logits = model(
      input_ids = input_ids,
      lengths=lengths
    )
    
    loss = criterion(
      logits,
      labels
    )
    loss.backward()
    optimizer.step()
    
    total_loss+= loss.item()
    
    predictions = torch.argmax(
      logits,
      dim=1
    )
    total_correct+=(
      predictions==labels
    ).sum().item()
    
    total_examples+=labels.size(0)
    
    
  
  average_loss = total_loss /len(train_loader)
  accuracy = total_correct/total_examples
  return average_loss,accuracy




def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    dict[str, float],
    list[int],
    list[int],
]:

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                lengths=lengths,
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    average_loss = total_loss / len(data_loader)

    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
    )

    return (
        average_loss,
        metrics,
        all_labels,
        all_predictions,
    )