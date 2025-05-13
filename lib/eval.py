"""
Evaluation Library for AI vs Human Image Classification

This module provides functions for testing image classification models in our notebook that are commonly shared by our models,
with metrics and visualizations for model evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch.amp import autocast

from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score, 
   confusion_matrix, classification_report, roc_curve, roc_auc_score,
   precision_recall_curve, average_precision_score, mean_squared_error
)

def compute_metrics(true_labels, predictions, probabilities):
   metrics = {}
   
   metrics['accuracy'] = accuracy_score(true_labels, predictions)
   metrics['precision'] = precision_score(true_labels, predictions, zero_division=0)
   metrics['recall'] = recall_score(true_labels, predictions, zero_division=0)
   metrics['f1'] = f1_score(true_labels, predictions, zero_division=0)
   
   metrics['rmse'] = np.sqrt(mean_squared_error(true_labels, probabilities))
   metrics['auc'] = roc_auc_score(true_labels, probabilities)
   metrics['avg_precision'] = average_precision_score(true_labels, probabilities)
   
   metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions)
   
   metrics['report'] = classification_report(true_labels, predictions, 
                                            target_names=['Human', 'AI-generated'])
   
   return metrics

def validate(model, val_loader, criterion, device):
   model.eval()
   total_loss = 0
   all_predictions = []
   all_labels = []
   all_probs = []
   
   with torch.no_grad():
       for images, labels in val_loader:
           images = images.to(device, non_blocking=True)
           labels = labels.float().to(device, non_blocking=True).unsqueeze(1)
           
           with autocast('cuda'):
               outputs = model(images)
               loss = criterion(outputs, labels)
           
           total_loss += loss.item() * images.size(0)
           
           probs = torch.sigmoid(outputs).cpu().numpy()
           preds = (probs >= 0.5).astype(int)
           
           all_predictions.extend(preds.flatten().tolist())
           all_labels.extend(labels.cpu().numpy().flatten().tolist())
           all_probs.extend(probs.flatten().tolist())
   
   metrics = compute_metrics(all_labels, all_predictions, all_probs)
   metrics['loss'] = total_loss / len(val_loader.dataset)
   
   fpr, tpr, _ = roc_curve(all_labels, all_probs)
   precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
   
   metrics['fpr'] = fpr
   metrics['tpr'] = tpr
   metrics['precision_curve'] = precision_curve
   metrics['recall_curve'] = recall_curve
   
   metrics['predictions'] = all_predictions
   metrics['labels'] = all_labels
   metrics['probabilities'] = all_probs
   
   return metrics

def evaluate_test_set(model, test_loader, criterion, device, save_path=None):
   model.eval()
   print("\nEvaluating on test set...")
   
   total_loss = 0
   all_predictions = []
   all_labels = []
   all_probs = []
   
   with torch.no_grad():
       for images, labels in test_loader:
           images = images.to(device, non_blocking=True)
           labels = labels.float().to(device, non_blocking=True).unsqueeze(1)
           
           with autocast('cuda'):
               outputs = model(images)
               loss = criterion(outputs, labels)
           
           total_loss += loss.item() * images.size(0)
           
           probs = torch.sigmoid(outputs).cpu().numpy()
           preds = (probs >= 0.5).astype(int)
           
           all_predictions.extend(preds.flatten().tolist())
           all_labels.extend(labels.cpu().numpy().flatten().tolist())
           all_probs.extend(probs.flatten().tolist())
   
   metrics = compute_metrics(all_labels, all_predictions, all_probs)
   metrics['loss'] = total_loss / len(test_loader.dataset)
   
   fpr, tpr, _ = roc_curve(all_labels, all_probs)
   precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
   
   metrics['fpr'] = fpr
   metrics['tpr'] = tpr
   metrics['precision_curve'] = precision_curve
   metrics['recall_curve'] = recall_curve
   
   metrics['predictions'] = all_predictions
   metrics['labels'] = all_labels
   metrics['probabilities'] = all_probs
   
   print("\n===== Test Set Results =====")
   print(f"Loss: {metrics['loss']:.4f}")
   print(f"RMSE: {metrics['rmse']:.4f}")
   print(f"AUC: {metrics['auc']:.4f}")
   print(f"Average Precision: {metrics['avg_precision']:.4f}")
   
   print("\nClassification Report:")
   print(metrics['report'])
   
   if save_path:
       os.makedirs(save_path, exist_ok=True)
       
       metrics_df = pd.DataFrame({
           'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'RMSE', 'AUC', 'Avg Precision'],
           'Value': [
               metrics['loss'], metrics['accuracy'], metrics['precision'], 
               metrics['recall'], metrics['f1'], metrics['rmse'], 
               metrics['auc'], metrics['avg_precision']
           ]
       })
       metrics_file = os.path.join(save_path, "test_metrics.csv")
       metrics_df.to_csv(metrics_file, index=False)
       print(f"Saved test metrics to {metrics_file}")
       
       results_df = pd.DataFrame({
           'true_label': all_labels,
           'predicted_label': all_predictions,
           'probability': all_probs,
           'prediction': ['AI-generated' if p == 1 else 'Human' for p in all_predictions]
       })
       results_file = os.path.join(save_path, "test_predictions.csv")
       results_df.to_csv(results_file, index=False)
       print(f"Saved test predictions to {results_file}")
   
   visualize_metrics(metrics, "Test", save_path)
   
   return metrics

def visualize_metrics(metrics, set_name="Test", save_path=None):
   plt.figure(figsize=(10, 8))
   
   sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Human', 'AI-generated'],
               yticklabels=['Human', 'AI-generated'])
   plt.title(f'{set_name} Set Confusion Matrix')
   plt.xlabel('Predicted Label')
   plt.ylabel('True Label')
   
   if save_path:
       confusion_file = os.path.join(save_path, f"{set_name.lower()}_confusion_matrix.png")
       plt.savefig(confusion_file, bbox_inches='tight', dpi=300)
       print(f"Saved confusion matrix to {confusion_file}")
   
   plt.show()
   
   if 'fpr' in metrics and 'tpr' in metrics:
       fpr, tpr = metrics['fpr'], metrics['tpr']
   elif hasattr(metrics, 'get') and metrics.get('roc_curve') is not None:
       fpr, tpr, _ = metrics['roc_curve']
   else:
       if 'labels' in metrics and 'probabilities' in metrics:
           fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
       else:
           fpr, tpr = None, None
   
   if 'precision_curve' in metrics and 'recall_curve' in metrics:
       precision, recall = metrics['precision_curve'], metrics['recall_curve']
   elif hasattr(metrics, 'get') and metrics.get('pr_curve') is not None:
       precision, recall, _ = metrics['pr_curve']
   else:
       if 'labels' in metrics and 'probabilities' in metrics:
           precision, recall, _ = precision_recall_curve(metrics['labels'], metrics['probabilities'])
       else:
           precision, recall = None, None
   
   plt.figure(figsize=(18, 6))
   
   plt.subplot(1, 3, 1)
   if fpr is not None and tpr is not None:
       plt.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
       plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.grid(alpha=0.3)
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title(f'{set_name} Set ROC Curve')
       plt.legend(loc="lower right")
   else:
       plt.text(0.5, 0.5, 'ROC data not available', ha='center', va='center')
       plt.title('ROC Curve (Not Available)')
   
   plt.subplot(1, 3, 2)
   if precision is not None and recall is not None:
       plt.plot(recall, precision, color='blue', lw=2, 
               label=f'AP = {metrics["avg_precision"]:.4f}')
       plt.grid(alpha=0.3)
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('Recall')
       plt.ylabel('Precision')
       plt.title(f'{set_name} Set Precision-Recall Curve')
       plt.legend(loc="lower left")
   else:
       plt.text(0.5, 0.5, 'PR curve data not available', ha='center', va='center')
       plt.title('Precision-Recall Curve (Not Available)')  
   if save_path:
       curves_file = os.path.join(save_path, f"{set_name.lower()}_curves.png")
       plt.savefig(curves_file, bbox_inches='tight', dpi=300)
       print(f"Saved curves to {curves_file}")
   
   plt.show()

def show_misclassified_examples(model, dataloader, num_examples=5, device=None):
   if device is None:
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   model.eval()
   misclassified = []
   
   with torch.no_grad():
       for images, labels in dataloader:
           images = images.to(device)
           
           if not isinstance(labels, torch.Tensor):
               print("Cannot show misclassified examples: DataLoader contains non-tensor labels")
               return
           
           labels = labels.to(device)
           
           outputs = model(images)
           preds = (torch.sigmoid(outputs) >= 0.5).squeeze().cpu().numpy()
           
           for i in range(len(images)):
               try:
                   if preds[i] != labels[i].cpu().numpy():
                       misclassified.append({
                           'image': images[i].cpu(),
                           'true_label': labels[i].item(),
                           'pred_label': preds[i].item() if isinstance(preds[i], np.ndarray) else preds[i],
                           'probability': torch.sigmoid(outputs[i]).item()
                       })
                       if len(misclassified) >= num_examples:
                           break
               except:
                   if preds != labels[i].cpu().numpy():
                       misclassified.append({
                           'image': images[i].cpu(),
                           'true_label': labels[i].item(),
                           'pred_label': preds if isinstance(preds, np.ndarray) else preds,
                           'probability': torch.sigmoid(outputs[i]).item()
                       })
                       if len(misclassified) >= num_examples:
                           break
           
           if len(misclassified) >= num_examples:
               break
   
   if not misclassified:
       print("No misclassified examples found in the provided data.")
       return
   
   fig, axes = plt.subplots(1, min(num_examples, len(misclassified)), figsize=(20, 4))
   
   if num_examples == 1:
       axes = [axes]
   
   mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
   std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
   
   for i, example in enumerate(misclassified[:num_examples]):
       img = example['image']
       img = img * std + mean
       img = img.permute(1, 2, 0).numpy()
       img = np.clip(img, 0, 1)
       
       axes[i].imshow(img)
       label_map = {0: 'Human', 1: 'AI-generated'}
       true_label = label_map[example['true_label']]
       pred_label = label_map[example['pred_label']]
       prob = example['probability']
       
       if example['pred_label'] == 1:
           conf_str = f"Prob: {prob:.3f}"
       else:
           conf_str = f"Prob: {1-prob:.3f}"
           
       axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\n{conf_str}", color='red')
       axes[i].axis('off')
   
   plt.suptitle("Misclassified Examples", fontsize=16)
   plt.tight_layout()
   plt.show()
   