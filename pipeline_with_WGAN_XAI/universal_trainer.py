"""
Universal Model Training System
Just drop models in models/ folder and run with one function call!

Usage:
    python train.py --model cnn_bilstm --epochs 50 --lr 0.001
    
Or in notebook:
    from universal_trainer import train_model
    train_model('cnn_bilstm', epochs=50, lr=0.001)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import importlib
import sys


# ============================================================================
# PART 1: Model Registry (Auto-discovers all models)
# ============================================================================

class ModelRegistry:
    """
    Automatically discovers and registers all models in models/ folder
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.registry = {}
        self._discover_models()
    
    def _discover_models(self):
        """Automatically find all model files"""
        print("🔍 Discovering models...")
        
        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return
        
        # Look for all Python files
        for model_file in self.models_dir.glob('*.py'):
            if model_file.stem.startswith('_'):
                continue
            
            try:
                # Import the module
                module_name = f"{self.models_dir.name}.{model_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, model_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Check if it has required attributes
                if hasattr(module, 'MODEL_CLASS') and hasattr(module, 'MODEL_CONFIG'):
                    model_name = model_file.stem
                    self.registry[model_name] = {
                        'class': module.MODEL_CLASS,
                        'config': module.MODEL_CONFIG,
                        'module': module
                    }
                    print(f"   ✅ Registered: {model_name}")
                
            except Exception as e:
                print(f"   ⚠️  Failed to load {model_file.name}: {str(e)[:50]}")
    
    def get_model(self, model_name):
        """Get model class and config"""
        if model_name not in self.registry:
            available = ', '.join(self.registry.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        return self.registry[model_name]
    
    def list_models(self):
        """List all available models"""
        print("\n📋 Available Models:")
        print("="*70)
        for name, info in self.registry.items():
            config = info['config']
            print(f"\n🔹 {name}")
            print(f"   Name: {config.get('name', 'N/A')}")
            print(f"   Description: {config.get('description', 'N/A')}")
            if 'reference' in config:
                print(f"   Reference: {config['reference']}")


# ============================================================================
# PART 2: Universal Dataset Wrapper
# ============================================================================

class ECGDataset(Dataset):
    """Universal ECG dataset that works with any model"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        if self.X.ndim == 2:
            self.X = self.X.unsqueeze(1)  # Add channel dimension
        
        # Convert labels to integers if they're strings
        if isinstance(y[0], str):
            label_map = {'N': 0, 'A': 1}
            y = np.array([label_map[label] for label in y])
        
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
# PART 3: Universal Training Function
# ============================================================================

def train_model(
    model_name,
    data_path='../data/processed/',
    epochs=50,
    lr=0.001,
    batch_size=32,
    device='auto',
    save_dir='../results/',
    early_stopping_patience=15,
    test_size=0.15,
    val_size=0.15,
    random_seed=42
):
    """
    Universal training function - works with ANY registered model!
    
    Args:
        model_name: Name of model file (e.g., 'cnn_bilstm')
        data_path: Path to processed data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: 'cuda', 'cpu', or 'auto'
        save_dir: Where to save results
        early_stopping_patience: Patience for early stopping
        test_size: Test set proportion
        val_size: Validation set proportion
        random_seed: Random seed for reproducibility
    
    Returns:
        results: Dictionary with training history and metrics
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 UNIVERSAL MODEL TRAINER")
    print(f"{'='*70}")
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Batch Size: {batch_size}")
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # ========================================================================
    # STEP 1: Load Model
    # ========================================================================
    
    print(f"\n📦 Loading model...")
    registry = ModelRegistry()
    model_info = registry.get_model(model_name)
    
    ModelClass = model_info['class']
    model_config = model_info['config'].copy()
    
    # Override config with function arguments
    model_config['learning_rate'] = lr
    model_config['batch_size'] = batch_size
    model_config['epochs'] = epochs
    
    model = ModelClass(model_config).to(device)
    
    print(f"   ✅ Model loaded: {model_config.get('name', model_name)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    
    print(f"\n📊 Loading data...")
    data_path = Path(data_path)
    
    X = np.load(data_path / 'X_combined.npy')
    y = np.load(data_path / 'y_combined.npy')
    
    print(f"   Total samples: {len(X):,}")
    print(f"   Signal shape: {X.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_seed
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_size / (test_size + val_size)), 
        stratify=y_temp, random_state=random_seed
    )
    
    print(f"\n   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ========================================================================
    # STEP 3: Setup Training
    # ========================================================================
    
    print(f"\n🔧 Setting up training...")
    
    # Loss function (check if model has custom loss)
    if hasattr(model_info['module'], 'LOSS_FUNCTION'):
        criterion = model_info['module'].LOSS_FUNCTION()
        print(f"   Using custom loss: {criterion.__class__.__name__}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"   Using default loss: CrossEntropyLoss")
    
    # Optimizer (check if model has custom optimizer settings)
    optimizer_config = model_config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'Adam')
    
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=optimizer_config.get('weight_decay', 1e-5)
        )
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 1e-5)
        )
    
    print(f"   Optimizer: {optimizer_type}")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    
    # ========================================================================
    # STEP 4: Training Loop
    # ========================================================================
    
    print(f"\n🏋️  Training...")
    print(f"{'='*70}\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auroc': [],
        'learning_rates': []
    }
    
    best_val_auroc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
        for ecg, labels in pbar:
            ecg, labels = ecg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (handle models with/without auxiliary output)
            outputs = model(ecg)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for ecg, labels in tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [Val]  ', leave=False):
                ecg, labels = ecg.to(device), labels.to(device)
                
                outputs = model(ecg)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        epoch_val_f1 = f1_score(val_labels, val_preds)
        epoch_val_auroc = roc_auc_score(val_labels, val_probs)
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)
        history['val_auroc'].append(epoch_val_auroc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        print(f"  Val F1:     {epoch_val_f1:.4f} | Val AUROC: {epoch_val_auroc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_auroc)
        
        # Save best model
        if epoch_val_auroc > best_val_auroc:
            best_val_auroc = epoch_val_auroc
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f"  ✅ New best model! (AUROC: {best_val_auroc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            break
    
    # ========================================================================
    # STEP 5: Test Evaluation
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"📊 FINAL EVALUATION ON TEST SET")
    print(f"{'='*70}\n")
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for ecg, labels in tqdm(test_loader, desc='Testing'):
            ecg = ecg.to(device)
            
            outputs = model(ecg)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate final metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_auroc = roc_auc_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"Test Results:")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  F1-Score:    {test_f1:.4f}")
    print(f"  AUROC:       {test_auroc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
    print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
    
    # ========================================================================
    # STEP 6: Save Results
    # ========================================================================
    
    save_dir = Path(save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'config': model_config,
        'history': history,
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_score': float(test_f1),
            'auroc': float(test_auroc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity)
        }
    }, save_dir / 'best_model.pth')
    
    # Save results JSON
    results = {
        'model_name': model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': model_config,
        'history': history,
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_score': float(test_f1),
            'auroc': float(test_auroc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist()
        }
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {save_dir}")
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}\n")
    
    return results


# ============================================================================
# PART 4: Command Line Interface
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Model Trainer')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--list', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list:
        registry = ModelRegistry()
        registry.list_models()
    else:
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device
        )