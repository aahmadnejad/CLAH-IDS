#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import gc
from datetime import datetime
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientTrainer:
    def __init__(self, output_dir="Final_Results"):
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        
    def setup_directories(self):
        dirs = [
            'models', 'metrics', 'plots', 'logs', 'explainability',
            'explainability/shap', 'explainability/lime', 
            'explainability/attention', 'data_analysis'
        ]
        for dir_name in dirs:
            os.makedirs(os.path.join(self.output_dir, dir_name), exist_ok=True)
        
        log_file = os.path.join(self.output_dir, 'logs', 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
    def load_data_efficiently(self, data_dir):
        logger.info("Starting memory-efficient data loading")
        
        attack_types = [
            'ARP_Spoof', 'ARP_poisoning', 'CAM_Table_flood', 
            'CDP_attack', 'DHCP_spoof', 'DHCP_starv',
            'Impersonation_attack', 'STP_attack', 'Switch_Spoof', 'VLAN_attack'
        ]
        
        chunk_size = 50000
        all_data = []
        
        for attack_type in attack_types:
            attack_path = os.path.join(data_dir, attack_type)
            if not os.path.exists(attack_path):
                logger.warning(f"Attack type directory not found: {attack_path}")
                continue
                
            logger.info(f"Processing {attack_type}...")
            
            for file_name in os.listdir(attack_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(attack_path, file_name)
                    
                    try:
                        chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
                        
                        for chunk_idx, chunk in enumerate(chunk_iterator):
                            chunk = chunk.replace(-100, np.nan)
                            chunk = chunk.fillna(chunk.median(numeric_only=True))
                            chunk = chunk.fillna(0)
                            
                            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                            if 'label' in chunk.columns and 'label' in numeric_cols:
                                numeric_cols = numeric_cols.drop('label')
                            
                            chunk_features = chunk[numeric_cols].values
                            chunk_labels = [attack_type] * len(chunk)
                            
                            all_data.append({
                                'features': chunk_features,
                                'labels': chunk_labels,
                                'attack_type': attack_type,
                                'file': file_name,
                                'chunk': chunk_idx
                            })
                            
                            del chunk
                            gc.collect()
                            
                            if len(all_data) % 10 == 0:
                                logger.info(f"Processed {len(all_data)} chunks")
                    
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue
        
        logger.info(f"Loaded {len(all_data)} data chunks")
        return all_data
    
    def process_benign_data(self, data_dir, sample_ratio=0.3):
        logger.info("Processing benign data with sampling")
        
        benign_path = os.path.join(data_dir, 'Benign')
        if not os.path.exists(benign_path):
            logger.warning("Benign data directory not found")
            return []
        
        benign_data = []
        chunk_size = 50000
        
        for file_name in os.listdir(benign_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(benign_path, file_name)
                
                try:
                    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
                    
                    for chunk_idx, chunk in enumerate(chunk_iterator):
                        sampled_chunk = chunk.sample(frac=sample_ratio, random_state=42)
                        
                        sampled_chunk = sampled_chunk.replace(-100, np.nan)
                        sampled_chunk = sampled_chunk.fillna(sampled_chunk.median(numeric_only=True))
                        sampled_chunk = sampled_chunk.fillna(0)
                        
                        numeric_cols = sampled_chunk.select_dtypes(include=[np.number]).columns
                        if 'label' in sampled_chunk.columns and 'label' in numeric_cols:
                            numeric_cols = numeric_cols.drop('label')
                        
                        chunk_features = sampled_chunk[numeric_cols].values
                        chunk_labels = ['Benign'] * len(sampled_chunk)
                        
                        benign_data.append({
                            'features': chunk_features,
                            'labels': chunk_labels,
                            'attack_type': 'Benign',
                            'file': file_name,
                            'chunk': chunk_idx
                        })
                        
                        del chunk, sampled_chunk
                        gc.collect()
                
                except Exception as e:
                    logger.error(f"Error processing benign file {file_path}: {e}")
                    continue
        
        logger.info(f"Processed {len(benign_data)} benign chunks")
        return benign_data
    
    def combine_data_efficiently(self, attack_data, benign_data, max_samples_per_type=100000):
        logger.info("Combining attack and benign data")
        
        combined_features = []
        combined_labels = []
        
        attack_type_counts = {}
        for data_chunk in attack_data:
            attack_type = data_chunk['attack_type']
            if attack_type not in attack_type_counts:
                attack_type_counts[attack_type] = 0
            
            if attack_type_counts[attack_type] < max_samples_per_type:
                combined_features.append(data_chunk['features'])
                combined_labels.extend(data_chunk['labels'])
                attack_type_counts[attack_type] += len(data_chunk['features'])
        
        benign_count = 0
        for data_chunk in benign_data:
            if benign_count < max_samples_per_type * 2:
                combined_features.append(data_chunk['features'])
                combined_labels.extend(data_chunk['labels'])
                benign_count += len(data_chunk['features'])
        
        X = np.vstack(combined_features)
        y = np.array(combined_labels)
        
        logger.info(f"Final dataset shape: {X.shape}")
        logger.info(f"Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"  {cls}: {count:,} samples")
        
        del combined_features
        gc.collect()
        
        return X, y
    
    def create_data_splits(self, X, y, test_size=0.2, val_size=0.2):
        logger.info("Creating data splits")
        
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        logger.info(f"Train: {X_train.shape[0]:,} samples")
        logger.info(f"Validation: {X_val.shape[0]:,} samples") 
        logger.info(f"Test: {X_test.shape[0]:,} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder
    
    def train_efficient_model(self, train_data, val_data, test_data, label_encoder):
        logger.info("Starting efficient model training")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Training Random Forest model")
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_val_scaled, y_val)
        test_score = model.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Validation accuracy: {val_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        return model, scaler, {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'test_accuracy': test_score,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'label_encoder': label_encoder
        }
    
    def evaluate_model(self, model, evaluation_data):
        logger.info("Performing comprehensive model evaluation")
        
        X_test = evaluation_data['X_test']
        y_test = evaluation_data['y_test']
        label_encoder = evaluation_data['label_encoder']
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = (y_pred == y_test).mean()
        
        report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        logger.info(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        self.save_evaluation_results(report, cm, label_encoder.classes_)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_evaluation_results(self, report, cm, class_names):
        logger.info("Saving evaluation results")
        
        with open(os.path.join(self.output_dir, 'metrics', 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        f1_scores = [report[cls]['f1-score'] for cls in class_names if cls in report]
        bars = plt.bar(class_names, f1_scores, color='skyblue', alpha=0.7)
        plt.title('F1-Score by Class')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'f1_scores.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_training(self, data_dir):
        logger.info("Starting complete memory-efficient training pipeline")
        
        try:
            attack_data = self.load_data_efficiently(data_dir)
            benign_data = self.process_benign_data(data_dir)
            
            X, y = self.combine_data_efficiently(attack_data, benign_data)
            
            del attack_data, benign_data
            gc.collect()
            
            train_data, val_data, test_data, label_encoder = self.create_data_splits(X, y)
            
            del X, y
            gc.collect()
            
            model, scaler, training_results = self.train_efficient_model(
                train_data, val_data, test_data, label_encoder
            )
            
            evaluation_results = self.evaluate_model(model, training_results)
            
            import pickle
            
            model_path = os.path.join(self.output_dir, 'models', 'trained_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            scaler_path = os.path.join(self.output_dir, 'models', 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            encoder_path = os.path.join(self.output_dir, 'models', 'label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            
            final_results = {
                'training_results': training_results,
                'evaluation_results': {
                    'accuracy': evaluation_results['accuracy'],
                    'weighted_f1': evaluation_results['classification_report']['weighted avg']['f1-score'],
                    'macro_f1': evaluation_results['classification_report']['macro avg']['f1-score']
                },
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'scaler_path': scaler_path,
                'encoder_path': encoder_path
            }
            
            with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info("Training completed successfully")
            logger.info(f"Results saved in: {self.output_dir}")
            logger.info(f"Final accuracy: {evaluation_results['accuracy']:.4f}")
            logger.info(f"Final weighted F1: {evaluation_results['classification_report']['weighted avg']['f1-score']:.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    data_dir = "../RESULTS/labeled_datasets"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    trainer = MemoryEfficientTrainer("Efficient_Training_Results")
    results = trainer.run_complete_training(data_dir)
    
    logger.info("Memory-efficient training completed successfully")

if __name__ == "__main__":
    main()