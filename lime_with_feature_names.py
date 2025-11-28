#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    logger.info("LIME library available")
except ImportError:
    logger.warning("LIME not available. Install with: pip install lime")
    LIME_AVAILABLE = False

class CompactIDSModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, window_size=30):
        super().__init__()
        
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        
        self.lstm = torch.nn.LSTM(128, 64, batch_first=True, dropout=0.2)
        self.attention = torch.nn.Linear(64, 1)
        self.binary_head = torch.nn.Linear(64, 2)
        self.multiclass_head = torch.nn.Linear(64, num_classes)
    
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

class EnhancedLIMEAnalyzer:
    def __init__(self, model_path, feature_names_csv=None, window_size=30):
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(self.num_classes)])
        
        self.model = CompactIDSModel(self.input_dim, self.num_classes, window_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        if feature_names_csv and os.path.exists(feature_names_csv):
            sample_df = pd.read_csv(feature_names_csv)
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            self.feature_names = numeric_cols[:self.input_dim]
        else:
            self.feature_names = [f'feature_{i}' for i in range(self.input_dim)]
        
        logger.info(f"Model loaded - Features: {self.input_dim}, Classes: {self.num_classes}")
        logger.info(f"Using {len(self.feature_names)} feature names")
    
    def predict_proba_flattened(self, X_flat):
        if len(X_flat.shape) == 1:
            X_flat = X_flat.reshape(1, -1)
        
        X_reshaped = X_flat.reshape(X_flat.shape[0], self.window_size, self.input_dim)
        
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        
        with torch.no_grad():
            _, logits = self.model(X_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict_proba(self, X):
        if len(X.shape) == 2:
            X = torch.FloatTensor(X).unsqueeze(0)
        else:
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            _, logits = self.model(X)
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def load_sample_data(self, data_dir, max_samples=1000):
        logger.info("Loading sample data for LIME analysis...")
        
        all_samples = []
        
        for attack_dir in os.listdir(data_dir):
            attack_path = os.path.join(data_dir, attack_dir)
            if os.path.isdir(attack_path):
                csv_files = [f for f in os.listdir(attack_path) if f.endswith('.csv')]
                
                for csv_file in csv_files[:1]:
                    try:
                        file_path = os.path.join(attack_path, csv_file)
                        df = pd.read_csv(file_path)
                        
                        df = df.replace(-100, np.nan)
                        df = df.fillna(df.median(numeric_only=True))
                        df = df.fillna(0)
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        features_df = df[numeric_cols]
                        
                        if 'label' in features_df.columns:
                            features_df = features_df.drop('label', axis=1)
                        
                        sample_size = min(50, len(features_df))
                        sampled = features_df.sample(n=sample_size, random_state=42)
                        
                        all_samples.append(sampled.values)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_file}: {e}")
                        continue
        
        if all_samples:
            combined_samples = np.vstack(all_samples)
            if len(combined_samples) > max_samples:
                indices = np.random.choice(len(combined_samples), max_samples, replace=False)
                combined_samples = combined_samples[indices]
            
            logger.info(f"Loaded {len(combined_samples)} samples for analysis")
            return combined_samples
        else:
            logger.error("No samples could be loaded")
            return None
    
    def run_lime_analysis(self, X_sample, num_explanations=5):
        if not LIME_AVAILABLE:
            logger.error("LIME not available")
            return None
        
        logger.info("Running enhanced LIME analysis with feature names...")
        
        X_flat = X_sample.reshape(len(X_sample), -1)
        
        expanded_feature_names = []
        for t in range(self.window_size):
            for fname in self.feature_names:
                expanded_feature_names.append(f'{fname}_t{t}')
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_flat,
            feature_names=expanded_feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        explanations = []
        feature_importance_scores = {}
        
        for i in range(min(num_explanations, len(X_sample))):
            exp = explainer.explain_instance(
                X_flat[i],
                self.predict_proba_flattened,
                num_features=30
            )
            
            explanations.append({
                'sample_index': i,
                'explanation': exp.as_list(),
                'prediction_proba': self.predict_proba_flattened(X_flat[i:i+1])[0].tolist()
            })
            
            for feature_name, importance in exp.as_list():
                base_feature = feature_name.split('_t')[0] if '_t' in feature_name else feature_name
                if base_feature not in feature_importance_scores:
                    feature_importance_scores[base_feature] = []
                feature_importance_scores[base_feature].append(abs(importance))
        
        aggregated_importance = {
            feature: np.mean(scores) 
            for feature, scores in feature_importance_scores.items()
        }
        
        sorted_features = sorted(aggregated_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            'explanations': explanations,
            'feature_importance': aggregated_importance,
            'top_features': sorted_features[:20],
            'feature_names_used': self.feature_names
        }
    
    def visualize_feature_importance(self, lime_results, save_path='lime_feature_analysis.png'):
        if not lime_results:
            return
        
        logger.info("Creating feature importance visualization...")
        
        top_features = lime_results['top_features'][:15]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        features, importances = zip(*top_features)
        
        axes[0].barh(range(len(features)), importances, alpha=0.7, color='skyblue')
        axes[0].set_yticks(range(len(features)))
        axes[0].set_yticklabels(features)
        axes[0].set_xlabel('Average Absolute LIME Importance')
        axes[0].set_title('Top 15 Most Important Features (Aggregated)')
        axes[0].invert_yaxis()
        
        all_importances = list(lime_results['feature_importance'].values())
        axes[1].hist(all_importances, bins=20, alpha=0.7, color='lightcoral')
        axes[1].set_xlabel('LIME Importance Score')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title('Distribution of Feature Importance Scores')
        axes[1].axvline(np.mean(all_importances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_importances):.4f}')
        axes[1].legend()
        
        plt.suptitle('LIME Feature Analysis with Real Feature Names', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance visualization saved to {save_path}")
    
    def create_detailed_report(self, lime_results, save_dir='lime_analysis'):
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Creating detailed LIME report...")
        
        with open(os.path.join(save_dir, 'lime_feature_ranking_with_names.json'), 'w') as f:
            json.dump(lime_results, f, indent=2)
        
        self.visualize_feature_importance(
            lime_results, 
            os.path.join(save_dir, 'lime_comprehensive_analysis_with_names.png')
        )
        
        report_content = f"""# LIME Analysis Report with Real Feature Names

## Model Information
- Input Features: {self.input_dim}
- Window Size: {self.window_size}
- Number of Classes: {self.num_classes}
- Class Names: {', '.join(self.class_names)}

## Top 20 Most Important Features
"""
        
        for i, (feature, importance) in enumerate(lime_results['top_features'], 1):
            report_content += f"{i}. {feature}: {importance:.6f}\n"
        
        report_content += f"""

## Analysis Summary
- Total unique features analyzed: {len(lime_results['feature_importance'])}
- Number of explanations generated: {len(lime_results['explanations'])}
- Average feature importance: {np.mean(list(lime_results['feature_importance'].values())):.6f}
- Standard deviation: {np.std(list(lime_results['feature_importance'].values())):.6f}

## Feature Categories
"""
        
        feature_categories = {}
        for feature in lime_results['feature_importance'].keys():
            category = feature.split('_')[0] if '_' in feature else 'other'
            if category not in feature_categories:
                feature_categories[category] = 0
            feature_categories[category] += 1
        
        for category, count in sorted(feature_categories.items(), key=lambda x: x[1], reverse=True):
            report_content += f"- {category}: {count} features\n"
        
        with open(os.path.join(save_dir, 'lime_analysis_report.md'), 'w') as f:
            f.write(report_content)
        
        logger.info(f"Detailed LIME report saved in {save_dir}/")
        return lime_results

def main():
    model_path = 'BEST_TRAINED_MODEL.pt'
    data_dir = '../RESULTS/labeled_datasets'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    sample_csv = None
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    sample_csv = os.path.join(root, file)
                    break
            if sample_csv:
                break
    
    analyzer = EnhancedLIMEAnalyzer(model_path, sample_csv)
    
    sample_data = analyzer.load_sample_data(data_dir)
    if sample_data is not None:
        sample_windows = []
        for i in range(len(sample_data) - analyzer.window_size + 1):
            window = sample_data[i:i+analyzer.window_size]
            sample_windows.append(window)
        
        sample_windows = np.array(sample_windows[:50])
        
        lime_results = analyzer.run_lime_analysis(sample_windows)
        
        if lime_results:
            analyzer.create_detailed_report(lime_results)
            logger.info("Enhanced LIME analysis completed successfully")
        else:
            logger.error("LIME analysis failed")
    else:
        logger.error("Could not load sample data")

if __name__ == "__main__":
    main()