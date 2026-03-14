#!/usr/bin/env python
# ============================================================================
# AFib-ResLSTM Training Script
# Generated from 04_model_architecture.ipynb
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your model (adjust path as needed)
# from models.model_architecture import AFibResLSTM, ModelConfig, FocalLoss

# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    # Data paths
    DATA_DIR = Path('../data/processed/')
    MODEL_DIR = Path('../models/checkpoints/')
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42

# ============================================================================
# Dataset Class
# ============================================================================

class ECGDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ecg = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            ecg = self.transform(ecg)
        
        return ecg, label

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for ecg, labels in tqdm(dataloader, desc='Training'):
        ecg, labels = ecg.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(ecg)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for ecg, labels in tqdm(dataloader, desc='Validation'):
            ecg, labels = ecg.to(device), labels.to(device)
            
            logits, _ = model(ecg)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_auroc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_auroc

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    # Set seed
    torch.manual_seed(TrainingConfig.SEED)
    np.random.seed(TrainingConfig.SEED)
    
    # Load data
    print("Loading data...")
    X = np.load(TrainingConfig.DATA_DIR / 'X_combined.npy')
    y = np.load(TrainingConfig.DATA_DIR / 'y_combined.npy')
    
    # Convert labels to binary
    y = (y == 'A').astype(int)  # 1 for AFib, 0 for Normal
    
    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=TrainingConfig.SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=TrainingConfig.SEED
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, 
                            shuffle=False, num_workers=4)
    
    # Create model (uncomment after importing model classes)
    # model = AFibResLSTM(ModelConfig()).to(TrainingConfig.DEVICE)
    # criterion = FocalLoss(alpha=0.65, gamma=2.0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
    #                                                         patience=5, factor=0.5)
    
    print("Training script template ready!")
    print("Uncomment model creation lines and run to start training.")

if __name__ == '__main__':
    main()
