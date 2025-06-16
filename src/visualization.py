import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix

class Visualizer:
    def __init__(self, save_dir: Path):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualization results
        """
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')  # 使用默认样式
        sns.set_theme()  # 使用seaborn的默认主题
    
    def plot_data_distribution(self, delay_values: np.ndarray, title: str = "Delay Distribution"):
        """Plot the distribution of delay values."""
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        sns.histplot(delay_values[delay_values != -1], bins=50)
        plt.title("Delay Distribution (excluding packet loss)")
        plt.xlabel("Delay (ms)")
        plt.ylabel("Count")
        
        # Plot packet loss distribution
        plt.subplot(1, 2, 2)
        packet_loss = (delay_values == -1).astype(int)
        sns.countplot(x=packet_loss)
        plt.title("Packet Loss Distribution")
        plt.xlabel("Packet Loss (1) / No Loss (0)")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_feature_importance(self, importances, feature_names):
        """绘制特征重要性图"""
        plt.figure(figsize=(10, 6))
        
        # 获取特征重要性
        if isinstance(importances, np.ndarray):
            importance_values = importances
        else:
            importance_values = importances.feature_importances_
            
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 绘制条形图
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(self.save_dir / 'feature_importance.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(self.save_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]], title: str = "Model Comparison"):
        """Plot comparison of model metrics."""
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())
        
        # Create DataFrame for plotting
        df = pd.DataFrame(metrics).T
        
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', width=0.8)
        plt.title(title)
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred, model_name='Model'):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(self.save_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]], title: str = "Training History"):
        """Plot training history for neural network."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_correlation_matrix(self, X: np.ndarray, feature_names: List[str], 
                              title: str = "Feature Correlation Matrix"):
        """Plot correlation matrix of features."""
        corr_matrix = np.corrcoef(X.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=feature_names, yticklabels=feature_names)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_prediction_distribution(self, y_pred, model_name='Model'):
        """绘制预测分布"""
        plt.figure(figsize=(8, 6))
        sns.histplot(y_pred, bins=50)
        plt.title(f'Prediction Distribution - {model_name}')
        plt.xlabel('Predicted Value')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(self.save_dir / f'prediction_distribution_{model_name.lower().replace(" ", "_")}.png')
        plt.close() 