import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any
import sys

class ImprovedMLP(nn.Module):
    def __init__(self, input_size):
        super(ImprovedMLP, self).__init__()
        self.network = nn.Sequential(
            # 第一层：增加神经元数量，使用LeakyReLU
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            # 第二层：保持较大规模
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            # 第三层：逐渐减小规模
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            # 第四层：进一步减小
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            # 输出层：使用sigmoid
            nn.Linear(32, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_model = None
        self.nn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> Dict[str, Any]:
        """
        Train a Random Forest classifier with optimized parameters.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            
        Returns:
            Dict[str, Any]: Training results and feature importance
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Calculate class weights
        n_samples = len(y_train)
        n_positives = sum(y_train)
        n_negatives = n_samples - n_positives
        
        # Use balanced class weights
        class_weights = {
            0: n_samples / (2 * n_negatives),
            1: n_samples / (2 * n_positives)
        }
        
        # Create and train the model with optimized parameters
        self.rf_model = RandomForestClassifier(
            n_estimators=200,          # Increased number of trees
            max_depth=10,              # Limit tree depth to prevent overfitting
            min_samples_split=5,       # Minimum samples required to split a node
            min_samples_leaf=2,        # Minimum samples required in a leaf node
            max_features='sqrt',       # Number of features to consider for best split
            class_weight=class_weights,# Use balanced class weights
            random_state=self.random_state,
            n_jobs=-1                  # Use all available cores
        )
        self.rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.rf_model.predict(X_test)
        y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print detailed results
        print("\nModel Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print("True Negatives:", cm[0, 0])
        print("False Positives:", cm[0, 1])
        print("False Negatives:", cm[1, 0])
        print("True Positives:", cm[1, 1])
        
        # Print prediction distribution
        print("\nPrediction Distribution:")
        print(f"Predicted positive: {sum(y_pred)}")
        print(f"Predicted negative: {len(y_pred) - sum(y_pred)}")
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("--------------------------------------")
        print("\nTop 10 Most Important Features are:")
        # for i in range(min(10, len(importances))):
        #     print(f"feature_{indices[i]}: {importancess[indices[i]]:.4f}")

        for i in range(min(10, len(importances))):
            if feature_names is not None:
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            else:
                print(f"feature_{indices[i]}: {importances[indices[i]]:.4f}")
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm
        }
    
    def build_neural_network(self, input_dim: int) -> None:
        """
        Build a neural network model.
        
        Args:
            input_dim (int): Input dimension
        """
        self.nn_model = ImprovedMLP(input_dim).to(self.device)
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Build the model if not already built
        if self.nn_model is None:
            self.build_neural_network(X.shape[1])
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.nn_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        self.nn_model.train()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
        
        # Evaluation
        self.nn_model.eval()
        with torch.no_grad():
            y_pred = self.nn_model(X_test_tensor).cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
        
        return {
            'metrics': metrics
        }
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both models and return their metrics.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for both models
        """
        rf_results = self.train_random_forest(X, y)
        nn_results = self.train_neural_network(X, y)
        
        return {
            'random_forest': rf_results['metrics'],
            'neural_network': nn_results['metrics']
        }

def train_neural_network(X_train, y_train, X_test, y_test, epochs=500, batch_size=32, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 数据预处理
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = ImprovedMLP(X_train.shape[1]).to(device)
    
    # 计算类别权重 - 显著增加正类权重
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean() * 5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 使用AdamW优化器，添加L2正则化
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )

    # 记录训练历史
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rates': []
    }

    best_val_f1 = 0
    best_model = None
    patience = 50
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
            # 计算训练准确率
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # 计算验证集性能
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
            val_correct = (val_predicted == y_test).sum().item()
            val_total = y_test.size(0)
            
            # 计算验证集指标
            val_pred = val_predicted.cpu().numpy()
            val_true = y_test.cpu().numpy()
            val_f1 = f1_score(val_true, val_pred)
            val_auc = roc_auc_score(val_true, torch.sigmoid(val_outputs).cpu().numpy())
            val_precision = precision_score(val_true, val_pred)
            val_recall = recall_score(val_true, val_pred)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"\nLearning rate decreased from {old_lr:.6f} to {new_lr:.6f}")
        
        # 记录历史
        avg_loss = epoch_loss / X_train.size()[0]
        train_accuracy = correct / total
        val_accuracy = val_correct / val_total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss.item())
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['learning_rates'].append(new_lr)
        
        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print(f"\rEpoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | "
              f"Val Prec: {val_precision:.4f} | Val Rec: {val_recall:.4f}", end="")
    
    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_pred_label = (torch.sigmoid(torch.tensor(y_pred)) > 0.5).numpy()
        print("\n\nFinal Test Results:")
        print("Accuracy:", accuracy_score(y_test.cpu(), y_pred_label))
        print("Precision:", precision_score(y_test.cpu(), y_pred_label))
        print("Recall:", recall_score(y_test.cpu(), y_pred_label))
        print("F1 Score:", f1_score(y_test.cpu(), y_pred_label))
        print("ROC AUC:", roc_auc_score(y_test.cpu(), torch.sigmoid(torch.tensor(y_pred)).numpy()))

    return model, history 