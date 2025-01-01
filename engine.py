import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.optim import lr_scheduler
from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix
import torch 
import os
from pathlib import Path
import shutil
import random
import json
from model import save_model

def train_step(model:nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device)-> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """validate a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a valing dataset.

  Args:
    model: A PyTorch model to be valed.
    dataloader: A DataLoader instance for the model to be valed on.
    loss_fn: A PyTorch loss function to calculate loss on the val data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of valing loss and valing accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval()

  # Setup val loss and val accuracy values
  val_loss, val_acc = 0, 0


  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          val_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(val_pred_logits, y)
          val_loss += loss.item()

          # Calculate and accumulate accuracy
          val_pred_labels = val_pred_logits.argmax(dim=1)
          val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))



  # Adjust metrics to get average loss and accuracy per batch
  val_loss = val_loss / len(dataloader)
  val_acc = val_acc / len(dataloader)
  return val_loss, val_acc


def eval_model(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
               num_classes:int,
              device: torch.device):
  
  """evaluate a pytorch model for a test set.

Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    num_classes: A integer that shows how many class dataset has.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary like below.
        {"test_loss":[],
           "test_acc":[],
           "test_precision":[],
           "test_recall":[],
           "test_f1_score":[],
           "test_confusion_matrix":[]}

  """
  test_acc,test_loss=0,0  
  model.eval()

  # Setup test loss and test accuracy values
  results={"test_loss":[],
           "test_acc":[],
           "test_precision":[],
           "test_recall":[],
           "test_f1_score":[],
           "test_confusion_matrix":[]}

  precision_metric = Precision(num_classes=num_classes,task="multiclass").to(device)
  recall_metric = Recall(num_classes=num_classes,task="multiclass").to(device)
  f1_score_metric = F1Score(num_classes=num_classes,task="multiclass").to(device)
  confusion_matrix = ConfusionMatrix(num_classes=num_classes,task="multiclass").to(device)


  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

          precision_metric(test_pred_labels, y)
          recall_metric(test_pred_labels, y)
          f1_score_metric(test_pred_labels, y)
          confusion_matrix(test_pred_labels, y)



  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)

  final_precision = precision_metric.compute()
  final_recall = recall_metric.compute()
  final_f1 = f1_score_metric.compute()
  final_confusion_matrix = confusion_matrix.compute()

  results["test_loss"].append(test_loss)
  results["test_acc"].append(test_acc)
  results["test_precision"].append(final_precision)
  results["test_recall"].append(final_recall)
  results["test_f1_score"].append(final_f1)
  results["test_confusion_matrix"].append(final_confusion_matrix)


  return results


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          model_name:str,
          model_saving_dir:str) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  val_loss: [...],
                  val_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  val_loss: [1.2641, 1.5706],
                  val_acc: [0.3400, 0.2973]}
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      val_loss, val_acc = val_step(model=model,
          dataloader=val_dataloader,
          loss_fn=loss_fn,
          device=device)

      if (epoch)%10==0:
        save_model(model=model,
                   target_dir=model_saving_dir,
                   model_name=f"{epoch}_{model_name}")

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)

  # Return the filled results at the end of the epochs
  return results

