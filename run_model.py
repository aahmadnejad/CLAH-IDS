#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import logging
import warnings
from datetime import datetime
import gc

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientDataset(Dataset):
    def __init__(self, features, labels, window_size=30):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.num_windows = max(0, len(features) - window_size + 1)
        
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        window = self.features[idx:idx + self.window_size]
        label = self.labels[idx + self.window_size - 1]
        
        if len(window) < self.window_size:
            padding = np.tile(window[-1], (self.window_size - len(window), 1))
            window = np.vstack([window, padding])
        
        return torch.FloatTensor(window), torch.LongTensor([label])

class CompactIDSModel(nn.Module):
    def __init__(self, input_dim, num_classes, window_size=30):
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.lstm = nn.LSTM(128, 64, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(64, 1)
        self.binary_head = nn.Linear(64, 2)
        self.multiclass_head = nn.Linear(64, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, return_attention=False):
        batch_size, window_size, features = x.shape
        
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn_layers(x_cnn)
        
        lstm_input = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)
        
        attention_scores = torch.sigmoid(self.attention(lstm_out))
        attended_features = torch.sum(lstm_out * attention_scores, dim=1)
        
        binary_logits = self.binary_head(attended_features)
        multiclass_logits = self.multiclass_head(attended_features)
        
        if return_attention:
            return binary_logits, multiclass_logits, attention_scores
        
        return binary_logits, multiclass_logits

class MemoryEfficientTrainer:
    def __init__(self, device='cuda', window_size=30):
        self.device = device
        self.window_size = window_size
        logger.info(f"Memory-Efficient Trainer initialized - Device: {device}")
    
    def load_balanced_sample(self, data_dir, max_samples_per_class=8000):
        logger.info("Loading balanced sample from network data...")
        
        target_distribution = {
            'Benign': max_samples_per_class * 2,
            'ARP_Spoof': max_samples_per_class,
            'Switch_Spoof': max_samples_per_class,
            'ARP_poisoning': max_samples_per_class // 2,
            'Impersonation_attack': max_samples_per_class,
            'CAM_Table_flood': max_samples_per_class,
            'VLAN_attack': max_samples_per_class // 2,
            'CDP_attack': max_samples_per_class,
            'DHCP_spoof': max_samples_per_class,
            'DHCP_starv': max_samples_per_class,
            'STP_attack': max_samples_per_class // 4
        }
        
        all_features = []
        all_labels = []
        
        for attack_type, target_samples in target_distribution.items():
            attack_path = os.path.join(data_dir, attack_type)
            if not os.path.exists(attack_path):
                continue
                
            logger.info(f"Loading {target_samples} samples from {attack_type}...")
            
            attack_data = []
            csv_files = [f for f in os.listdir(attack_path) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(attack_path, csv_file)
                    df = pd.read_csv(file_path)
                    
                    df = df.replace(-100, np.nan)
                    df = df.fillna(df.median(numeric_only=True))
                    df = df.fillna(0)
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if 'label' in df.columns:
                        df = df[numeric_cols]
                    else:
                        df = df[numeric_cols]
                    
                    attack_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
                    continue
            
            if attack_data:
                combined_df = pd.concat(attack_data, ignore_index=True)
                
                if len(combined_df) > target_samples:
                    sampled_df = combined_df.sample(n=target_samples, random_state=42)
                else:
                    sampled_df = combined_df
                
                features = sampled_df.values
                labels = [attack_type] * len(sampled_df)
                
                all_features.append(features)
                all_labels.extend(labels)
                
                logger.info(f"  {attack_type}: {len(sampled_df)} samples loaded")
                
                del combined_df, sampled_df, attack_data
                gc.collect()
        
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        logger.info(f"Total dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Classes: {np.unique(y)}")
        
        return X, y
    
    def prepare_data(self, X, y):
        logger.info("Preparing data for training...")
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        
        chunk_size = 10000
        X_scaled = np.zeros_like(X)
        
        for i in range(0, len(X), chunk_size):
            end_idx = min(i + chunk_size, len(X))
            if i == 0:
                X_scaled[i:end_idx] = scaler.fit_transform(X[i:end_idx])
            else:
                X_scaled[i:end_idx] = scaler.transform(X[i:end_idx])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        logger.info("Applying class balancing...")
        
        smote_tomek = SMOTETomek(
            smote=SMOTE(random_state=42, k_neighbors=min(3, len(np.unique(y_train)) - 1)),
            tomek=TomekLinks()
        )
        
        X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
        
        logger.info(f"Final splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, scaler
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size=64):
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        train_dataset = MemoryEfficientDataset(X_train, y_train, self.window_size)
        val_dataset = MemoryEfficientDataset(X_val, y_val, self.window_size)
        test_dataset = MemoryEfficientDataset(X_test, y_test, self.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, num_classes, input_dim, epochs=50):
        logger.info("Training memory-efficient model...")
        
        model = CompactIDSModel(input_dim, num_classes, self.window_size).to(self.device)
        
        all_labels = []
        for _, batch_labels in train_loader:
            all_labels.extend(batch_labels.squeeze().numpy())
        
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        binary_labels = (np.array(all_labels) > 0).astype(int)
        binary_weights = compute_class_weight('balanced', classes=np.unique(binary_labels), y=binary_labels)
        
        criterion_multiclass = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        criterion_binary = nn.CrossEntropyLoss(weight=torch.FloatTensor(binary_weights).to(self.device))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_f1 = 0
        patience = 10
        patience_counter = 0
        accumulation_steps = 4
        
        training_history = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            optimizer.zero_grad()
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.squeeze().to(self.device)
                
                binary_targets = (targets > 0).long()
                
                binary_logits, multiclass_logits = model(data)
                
                binary_loss = criterion_binary(binary_logits, binary_targets)
                multiclass_loss = criterion_multiclass(multiclass_logits, targets)
                loss = 0.3 * binary_loss + 0.7 * multiclass_loss
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                predicted = torch.argmax(multiclass_logits, dim=1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(self.device)
                    targets = targets.squeeze().to(self.device)
                    
                    binary_targets = (targets > 0).long()
                    binary_logits, multiclass_logits = model(data)
                    
                    binary_loss = criterion_binary(binary_logits, binary_targets)
                    multiclass_loss = criterion_multiclass(multiclass_logits, targets)
                    loss = 0.3 * binary_loss + 0.7 * multiclass_loss
                    
                    val_loss += loss.item()
                    predicted = torch.argmax(multiclass_logits, dim=1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())
            
            train_acc = train_correct / train_total
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            scheduler.step()
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'lr': scheduler.get_last_lr()[0]
            }
            
            training_history.append(epoch_metrics)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}:")
                logger.info(f"  Train: Loss={epoch_metrics['train_loss']:.4f}, Acc={train_acc:.1%}")
                logger.info(f"  Val: Loss={epoch_metrics['val_loss']:.4f}, Acc={val_acc:.1%}, F1={val_f1:.4f}")
                logger.info(f"  LR: {epoch_metrics['lr']:.2e}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'window_size': self.window_size,
                    'best_f1': best_f1
                }, 'best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if epoch % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Training completed! Best F1: {best_f1:.4f}")
        
        return model, training_history, best_f1
    
    def evaluate_model(self, model, test_loader, class_names):
        logger.info("Evaluating model performance...")
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.squeeze()
                
                binary_logits, multiclass_logits = model(data)
                predicted = torch.argmax(multiclass_logits, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        
        cm = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"Final Test Results:")
        logger.info(f"   Accuracy: {accuracy:.1%}")
        logger.info(f"   F1 (Weighted): {f1_weighted:.4f}")
        logger.info(f"   F1 (Macro): {f1_macro:.4f}")
        
        logger.info("Per-class F1 scores:")
        for i, (class_name, f1) in enumerate(zip(class_names, per_class_f1)):
            logger.info(f"   {class_name}: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'per_class_f1': per_class_f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }

def main():
    logger.info("STARTING MEMORY-EFFICIENT IDS SYSTEM")
    logger.info("=" * 70)
    
    config = {
        'data_dir': '../RESULTS/labeled_datasets',
        'window_size': 30,
        'batch_size': 32,
        'epochs': 40,
        'max_samples_per_class': 6000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        trainer = MemoryEfficientTrainer(device=config['device'], window_size=config['window_size'])
        
        logger.info("Step 1: Loading balanced sample...")
        X, y = trainer.load_balanced_sample(config['data_dir'], config['max_samples_per_class'])
        
        logger.info("Step 2: Preparing data...")
        train_data, val_data, test_data, label_encoder, scaler = trainer.prepare_data(X, y)
        
        logger.info("Step 3: Creating data loaders...")
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            train_data, val_data, test_data, batch_size=config['batch_size']
        )
        
        class_names = label_encoder.classes_
        num_classes = len(class_names)
        input_dim = X.shape[1]
        
        logger.info(f"Classes: {list(class_names)}")
        logger.info(f"Features: {input_dim}, Classes: {num_classes}")
        
        logger.info("Step 4: Training model...")
        model, training_history, best_f1 = trainer.train_model(
            train_loader, val_loader, num_classes, input_dim, epochs=config['epochs']
        )
        
        logger.info("Step 5: Evaluating model...")
        results = trainer.evaluate_model(model, test_loader, class_names)
        
        logger.info("Step 6: Saving results...")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'num_classes': num_classes,
            'window_size': config['window_size'],
            'class_names': list(class_names),
            'best_f1': best_f1
        }, 'trained_model.pt')
        
        import pickle
        with open('preprocessors.pkl', 'wb') as f:
            pickle.dump({'scaler': scaler, 'label_encoder': label_encoder}, f)
        
        final_results = {
            'config': config,
            'training_history': training_history,
            'test_results': {
                'accuracy': results['accuracy'],
                'f1_weighted': results['f1_weighted'],
                'f1_macro': results['f1_macro'],
                'per_class_f1': results['per_class_f1'].tolist(),
                'class_names': list(class_names)
            },
            'model_info': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'window_size': config['window_size'],
                'best_f1': best_f1
            }
        }
        
        with open('results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info("MEMORY-EFFICIENT IDS SYSTEM COMPLETED!")
        logger.info("=" * 70)
        logger.info(f"Final Results:")
        logger.info(f"   Accuracy: {results['accuracy']:.1%}")
        logger.info(f"   F1-Score (Weighted): {results['f1_weighted']:.4f}")
        logger.info(f"   F1-Score (Macro): {results['f1_macro']:.4f}")
        logger.info(f"   Classes >= 80% F1: {np.sum(results['per_class_f1'] >= 0.8)}/{len(class_names)}")
        logger.info("=" * 70)
        
        if results['f1_weighted'] >= 0.8:
            logger.info("TARGET ACHIEVED: 80%+ F1 Score!")
        else:
            logger.info(f"Target Progress: {results['f1_weighted']:.1%} (Target: 80%)")
        
    except Exception as e:
        logger.error(f"Error in memory-efficient training: {e}")
        raise

if __name__ == "__main__":
    main()