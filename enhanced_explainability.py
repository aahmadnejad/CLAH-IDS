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
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library available")
except ImportError:
    logger.warning("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_image import LimeImageExplainer
    from lime.lime_tabular import LimeTabularExplainer
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
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
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

class IDSExplainer:
    def __init__(self, model_path, window_size=30):
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
        
        logger.info(f"Model loaded - Features: {self.input_dim}, Classes: {self.num_classes}")
    
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
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def get_attention_weights(self, X):
        if len(X.shape) == 2:
            X = torch.FloatTensor(X).unsqueeze(0)
        else:
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            _, _, attention = self.model(X, return_attention=True)
        
        return attention.squeeze().cpu().numpy()
    
    def explain_with_shap(self, X_sample, background_size=100):
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return None
        
        logger.info("Running SHAP analysis...")
        
        background = X_sample[:background_size]
        
        def model_predict(data):
            return self.predict_proba(data)
        
        explainer = shap.Explainer(model_predict, background)
        shap_values = explainer(X_sample[:10])
        
        return shap_values
    
    def explain_with_lime(self, X_sample, feature_names=None):
        if not LIME_AVAILABLE:
            logger.error("LIME not available")
            return None
        
        logger.info("Running LIME analysis...")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.input_dim)]
        
        flattened_data = X_sample.reshape(len(X_sample), -1)
        
        explainer = LimeTabularExplainer(
            flattened_data,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
        def predict_fn(data):
            reshaped = data.reshape(len(data), self.window_size, self.input_dim)
            return self.predict_proba(reshaped)
        
        explanations = []
        for i in range(min(5, len(X_sample))):
            exp = explainer.explain_instance(
                flattened_data[i],
                predict_fn,
                num_features=20
            )
            explanations.append(exp)
        
        return explanations
    
    def create_attention_heatmap(self, X_sample, save_path='attention_heatmap.png'):
        logger.info("Creating attention heatmap...")
        
        attention_weights = []
        predictions = []
        
        for i in range(min(10, len(X_sample))):
            sample = X_sample[i:i+1]
            attention = self.get_attention_weights(sample)
            pred = self.predict(sample)[0]
            
            attention_weights.append(attention)
            predictions.append(pred)
        
        attention_matrix = np.array(attention_weights)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_matrix, 
                   annot=False,
                   cmap='viridis',
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title('Attention Weights Across Time Steps')
        plt.xlabel('Time Step')
        plt.ylabel('Sample')
        
        y_labels = [f'Sample {i} ({self.class_names[pred]})' 
                   for i, pred in enumerate(predictions)]
        plt.yticks(range(len(y_labels)), y_labels)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention heatmap saved to {save_path}")
        return attention_matrix
    
    def create_comprehensive_analysis(self, X_sample, save_dir='analysis_results'):
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Starting comprehensive explainability analysis...")
        
        results = {
            'model_info': {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'window_size': self.window_size,
                'class_names': self.class_names
            }
        }
        
        attention_matrix = self.create_attention_heatmap(
            X_sample, 
            os.path.join(save_dir, 'attention_heatmap.png')
        )
        results['attention_analysis'] = {
            'mean_attention': attention_matrix.mean(axis=0).tolist(),
            'std_attention': attention_matrix.std(axis=0).tolist()
        }
        
        if SHAP_AVAILABLE:
            try:
                shap_values = self.explain_with_shap(X_sample)
                if shap_values is not None:
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(shap_values[0], max_display=20)
                    plt.savefig(os.path.join(save_dir, 'shap_waterfall_example.png'), 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    results['shap_analysis'] = {
                        'available': True,
                        'top_features_mean': np.mean(shap_values.values, axis=0).tolist()
                    }
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                results['shap_analysis'] = {'available': False, 'error': str(e)}
        
        if LIME_AVAILABLE:
            try:
                feature_names = [f'feature_{i}' for i in range(self.input_dim)]
                lime_explanations = self.explain_with_lime(X_sample, feature_names)
                
                if lime_explanations:
                    lime_results = []
                    for i, exp in enumerate(lime_explanations):
                        exp_data = {
                            'sample_index': i,
                            'prediction': int(self.predict(X_sample[i:i+1])[0]),
                            'explanation': exp.as_list()
                        }
                        lime_results.append(exp_data)
                    
                    results['lime_analysis'] = {
                        'available': True,
                        'explanations': lime_results
                    }
                    
                    feature_importance = {}
                    for exp in lime_explanations:
                        for feature, importance in exp.as_list():
                            if feature in feature_importance:
                                feature_importance[feature].append(importance)
                            else:
                                feature_importance[feature] = [importance]
                    
                    avg_importance = {
                        feature: np.mean(importances) 
                        for feature, importances in feature_importance.items()
                    }
                    
                    sorted_features = sorted(avg_importance.items(), 
                                           key=lambda x: abs(x[1]), reverse=True)
                    
                    with open(os.path.join(save_dir, 'lime_feature_ranking_with_names.json'), 'w') as f:
                        json.dump(sorted_features[:20], f, indent=2)
                    
                    plt.figure(figsize=(12, 8))
                    features, importances = zip(*sorted_features[:20])
                    colors = ['red' if imp < 0 else 'green' for imp in importances]
                    
                    plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Average LIME Importance')
                    plt.title('Top 20 Features by LIME Importance')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'lime_comprehensive_analysis_with_names.png'), 
                              dpi=300, bbox_inches='tight')
                    plt.close()
            
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
                results['lime_analysis'] = {'available': False, 'error': str(e)}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].plot(results['attention_analysis']['mean_attention'])
        axes[0, 0].fill_between(range(len(results['attention_analysis']['mean_attention'])),
                               np.array(results['attention_analysis']['mean_attention']) - np.array(results['attention_analysis']['std_attention']),
                               np.array(results['attention_analysis']['mean_attention']) + np.array(results['attention_analysis']['std_attention']),
                               alpha=0.3)
        axes[0, 0].set_title('Attention Pattern Analysis')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Attention Weight')
        
        sample_predictions = [self.predict(X_sample[i:i+1])[0] for i in range(min(10, len(X_sample)))]
        class_distribution = np.bincount(sample_predictions, minlength=self.num_classes)
        
        axes[0, 1].bar(range(len(class_distribution)), class_distribution)
        axes[0, 1].set_title('Sample Class Distribution')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(self.class_names)))
        axes[0, 1].set_xticklabels(self.class_names, rotation=45)
        
        confidence_scores = []
        for i in range(min(10, len(X_sample))):
            proba = self.predict_proba(X_sample[i:i+1])[0]
            confidence_scores.append(np.max(proba))
        
        axes[1, 0].hist(confidence_scores, bins=10, alpha=0.7)
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].text(0.1, 0.9, f'Model Architecture: CNN-LSTM-Attention', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f'Input Features: {self.input_dim}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Window Size: {self.window_size}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Classes: {self.num_classes}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'SHAP Available: {SHAP_AVAILABLE}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'LIME Available: {LIME_AVAILABLE}', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Information')
        axes[1, 1].axis('off')
        
        plt.suptitle('Comprehensive Explainability Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comprehensive_explainability.png'), 
                  dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(os.path.join(save_dir, 'explainability_summary.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Comprehensive explainability analysis completed")
        logger.info(f"Results saved in {save_dir}/")
        
        return results

def main():
    model_path = 'BEST_TRAINED_MODEL.pt'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    explainer = IDSExplainer(model_path)
    
    dummy_data = np.random.randn(20, explainer.window_size, explainer.input_dim)
    
    results = explainer.create_comprehensive_analysis(dummy_data)
    
    logger.info("Explainability analysis completed successfully")

if __name__ == "__main__":
    main()