"""
Data Loading and Exploration Library for AI vs Human Image Classification

This module provides functions for loading, preprocessing, exploring, and visualizing
image datasets for AI vs human classification tasks.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class AIvsHumanDataset(Dataset):
   def __init__(self, dataframe, transform=None):
       self.dataframe = dataframe
       self.transform = transform
       
   def __len__(self):
       return len(self.dataframe)
   
   def __getitem__(self, idx):
       if 'file_name' in self.dataframe.columns:
           img_path = self.dataframe.iloc[idx]['file_name']
       elif 'id' in self.dataframe.columns:
           img_path = self.dataframe.iloc[idx]['id']
       else:
           raise ValueError("DataFrame must contain either 'file_name' or 'id' column")
       
       try:
           image = Image.open(img_path).convert('RGB')
       except Exception as e:
           print(f"Error loading image {img_path}: {e}")
           image = Image.new('RGB', (224, 224), (0, 0, 0))
       
       if self.transform:
           image = self.transform(image)
       
       if 'label' in self.dataframe.columns:
           label = self.dataframe.iloc[idx]['label']
           return image, label
       else:
           return image, img_path
# "A Survey on Image Data Augmentation for Deep Learning" by Shorten & Khoshgoftaar (2019):
# https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
def get_transforms(image_size=224):
   train_transform = transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   val_transform = transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   return train_transform, val_transform

def load_data(train_csv, test_csv, val_split=0.2, random_state=42):
   train_df = pd.read_csv(train_csv)
   print(f"Training data shape: {train_df.shape}")
   
   test_df = pd.read_csv(test_csv)
   print(f"Test data shape: {test_df.shape}")
   
   train_data, val_data = train_test_split(
       train_df, 
       test_size=val_split, 
       random_state=random_state, 
       stratify=train_df['label'] if 'label' in train_df.columns else None
   )
   
   train_data = train_data.reset_index(drop=True)
   val_data = val_data.reset_index(drop=True)
   
   print(f"Training data size: {len(train_data)}")
   print(f"Validation data size: {len(val_data)}")
   
   return train_data, val_data, test_df

def create_dataloaders(train_df, val_df, test_df, batch_size=32, image_size=224, num_workers=4):
    train_transform, test_transform = get_transforms(image_size)
    
    if 'id' in test_df.columns and 'file_name' not in test_df.columns:
        test_df['file_name'] = test_df['id'].copy()

    train_dataset = AIvsHumanDataset(train_df, transform=train_transform)
    val_dataset = AIvsHumanDataset(val_df, transform=test_transform)
    test_dataset = AIvsHumanDataset(test_df, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def set_random_seed(seed=42):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   os.environ['PYTHONHASHSEED'] = str(seed)
   print(f"Random seed set to {seed}")

def check_data_integrity(dataframe, image_col=None, sample_size=10):
   if image_col is None:
       if 'file_name' in dataframe.columns:
           image_col = 'file_name'
       elif 'id' in dataframe.columns:
           image_col = 'id'
       else:
           raise ValueError("Cannot determine image path column. Specify 'image_col'.")
   
   if len(dataframe) < sample_size:
       sample_size = len(dataframe)
   
   sample_indices = random.sample(range(len(dataframe)), sample_size)
   all_valid = True
   
   print(f"Checking {sample_size} random images for integrity...")
   for idx in sample_indices:
       img_path = dataframe.iloc[idx][image_col]
       try:
           if not os.path.exists(img_path):
               print(f"File not found: {img_path}")
               all_valid = False
               continue
           
           with Image.open(img_path) as img:
               img.verify() 
               
           print(f"{img_path} - Valid")
       except Exception as e:
           print(f"{img_path} - Error: {e}")
           all_valid = False
   
   if all_valid:
       print("All sampled images are valid.")
   else:
       print("Some images have integrity issues. Check the errors above.")
   
   return all_valid

def explore_dataset(dataframe, label_col='label', label_names=None):
   print(f"Dataset shape: {dataframe.shape}")
   print("\nFirst few rows:")
   print(dataframe.head())
   
   print("\nColumns:")
   print(dataframe.columns.tolist())
   
   print("\nMissing values:")
   missing = dataframe.isnull().sum()
   print(missing[missing > 0] if any(missing > 0) else "No missing values")
   
   if label_col in dataframe.columns:
       label_counts = dataframe[label_col].value_counts()
       print("\nLabel distribution:")
       
       if label_names:
           for label_val, count in label_counts.items():
               label_name = label_names.get(label_val, f"Class {label_val}")
               print(f"  {label_name}: {count} ({count/len(dataframe)*100:.1f}%)")
       else:
           print(label_counts)
       
       plt.figure(figsize=(10, 6))
       ax = sns.countplot(x=label_col, data=dataframe)
       plt.title('Distribution of Classes')
       
       if label_names:
           ax.set_xticklabels([label_names.get(i, f"Class {i}") for i in range(len(label_counts))])
       
       plt.ylabel('Count')
       plt.grid(axis='y', alpha=0.3)
       
       for p in ax.patches:
           ax.annotate(f"{p.get_height()}", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', 
                      xytext = (0, 5), textcoords = 'offset points')
       
       plt.show()

def display_images_by_class(dataframe, class_label, label_col='label', image_col=None, 
                         num_images=5, figsize=(20, 4)):
   if image_col is None:
       if 'file_name' in dataframe.columns:
           image_col = 'file_name'
       elif 'id' in dataframe.columns:
           image_col = 'id'
       else:
           raise ValueError("Cannot determine image path column. Specify 'image_col'.")
   
   filtered_df = dataframe[dataframe[label_col] == class_label]
   
   if len(filtered_df) == 0:
       print(f"No images found for class {class_label}")
       return
   
   samples = filtered_df.sample(min(num_images, len(filtered_df)))
   
   fig, axes = plt.subplots(1, len(samples), figsize=figsize)
   
   if len(samples) == 1:
       axes = [axes]
   
   for i, (idx, row) in enumerate(samples.iterrows()):
       img_path = row[image_col]
       try:
           img = Image.open(img_path)
           axes[i].imshow(img)
           axes[i].set_title(f"Class: {class_label}")
           axes[i].axis('off')
       except Exception as e:
           print(f"Error loading image {img_path}: {e}")
           axes[i].text(0.5, 0.5, f"Error loading\n{img_path}", ha='center', va='center')
           axes[i].axis('off')
   
   plt.tight_layout()
   plt.show()

def show_augmented_samples(dataloader, num_images=5, figsize=(20, 4)):
   images, labels = next(iter(dataloader))
   
   images = images[:num_images]
   labels = labels[:num_images]
   
   fig, axes = plt.subplots(1, num_images, figsize=figsize)
   
   if num_images == 1:
       axes = [axes]
   
   mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
   std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
   
   for i in range(num_images):
       img = images[i]
       img = img * std + mean
       img = img.permute(1, 2, 0).cpu().numpy()
       img = np.clip(img, 0, 1)
       
       axes[i].imshow(img)
       axes[i].set_title(f"Label: {labels[i].item()}")
       axes[i].axis('off')
   
   plt.tight_layout()
   plt.show()

def get_image_statistics(dataframe, image_col=None, sample_size=100):
   if image_col is None:
       if 'file_name' in dataframe.columns:
           image_col = 'file_name'
       elif 'id' in dataframe.columns:
           image_col = 'id'
       else:
           raise ValueError("Cannot determine image path column. Specify 'image_col'.")
   
   if len(dataframe) < sample_size:
       sample_size = len(dataframe)
   
   sample_indices = random.sample(range(len(dataframe)), sample_size)
   
   widths = []
   heights = []
   aspect_ratios = []
   
   print(f"Analyzing dimensions for {sample_size} random images...")
   for idx in sample_indices:
       img_path = dataframe.iloc[idx][image_col]
       try:
           with Image.open(img_path) as img:
               width, height = img.size
               aspect_ratio = width / height
               
               widths.append(width)
               heights.append(height)
               aspect_ratios.append(aspect_ratio)
       except Exception as e:
           print(f"Error processing {img_path}: {e}")
   
   stats_df = pd.DataFrame({
       'width': widths,
       'height': heights,
       'aspect_ratio': aspect_ratios
   })
   
   print("\nImage dimension statistics:")
   print(f"Width: min={stats_df['width'].min()}, max={stats_df['width'].max()}, mean={stats_df['width'].mean():.1f}")
   print(f"Height: min={stats_df['height'].min()}, max={stats_df['height'].max()}, mean={stats_df['height'].mean():.1f}")
   print(f"Aspect ratio: min={stats_df['aspect_ratio'].min():.2f}, max={stats_df['aspect_ratio'].max():.2f}, mean={stats_df['aspect_ratio'].mean():.2f}")
   
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
   sns.histplot(stats_df['width'], bins=20, kde=True, ax=axes[0])
   axes[0].set_title('Width Distribution')
   axes[0].set_xlabel('Width (pixels)')
   
   sns.histplot(stats_df['height'], bins=20, kde=True, ax=axes[1])
   axes[1].set_title('Height Distribution')
   axes[1].set_xlabel('Height (pixels)')
   
   sns.histplot(stats_df['aspect_ratio'], bins=20, kde=True, ax=axes[2])
   axes[2].set_title('Aspect Ratio Distribution')
   axes[2].set_xlabel('Aspect Ratio (width/height)')
   
   plt.tight_layout()
   plt.show()
   
   plt.figure(figsize=(10, 8))
   sns.scatterplot(x='width', y='height', data=stats_df, alpha=0.7)
   plt.title('Image Dimensions: Width vs Height')
   plt.xlabel('Width (pixels)')
   plt.ylabel('Height (pixels)')
   plt.grid(True, alpha=0.3)
   plt.show()
   
   return stats_df

def get_basic_data_summary(dataframe, label_col='label', image_col=None):
   if image_col is None:
       if 'file_name' in dataframe.columns:
           image_col = 'file_name'
       elif 'id' in dataframe.columns:
           image_col = 'id'
       else:
           raise ValueError("Cannot determine image path column. Specify 'image_col'.")
   
   print(f"Dataset shape: {dataframe.shape}")
   
   if label_col in dataframe.columns:
       unique_labels = dataframe[label_col].unique()
       print(f"Number of classes: {len(unique_labels)}")
       
       class_dist = dataframe[label_col].value_counts(normalize=True) * 100
       class_counts = dataframe[label_col].value_counts()
       
       print("\nClass distribution:")
       for label, percentage in class_dist.items():
           count = class_counts[label]
           print(f"  Class {label}: {count} images ({percentage:.1f}%)")
   
   missing = dataframe.isnull().sum()
   if any(missing > 0):
       print("\nMissing values:")
       print(missing[missing > 0])
   else:
       print("\nNo missing values found.")
   
   total_files = len(dataframe)
   print(f"\nChecking a few random files (out of {total_files} total):")
   
   for i, filename in enumerate(random.sample(dataframe[image_col].tolist(), min(5, total_files))):
       print(f"  {i+1}. {filename}: {'exists' if os.path.exists(filename) else 'MISSING'}")
   
   if label_col in dataframe.columns:
       for label in unique_labels:
           print(f"\nClass {label} examples:")
           display_images_by_class(dataframe, label, label_col, image_col, num_images=3, figsize=(15, 4))