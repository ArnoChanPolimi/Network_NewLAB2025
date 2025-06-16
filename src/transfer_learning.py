import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from src.model_training import ModelTrainer
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class TransferLearningTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_trainer = ModelTrainer(random_state=random_state)
        
    def prepare_transfer_data(self, 
                            source_data: Tuple[np.ndarray, np.ndarray],
                            target_data: Tuple[np.ndarray, np.ndarray],
                            test_size: float = 0.2) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data for transfer learning by splitting into train and test sets.
        
        Args:
            source_data (Tuple[np.ndarray, np.ndarray]): Source direction data (X, y)
            target_data (Tuple[np.ndarray, np.ndarray]): Target direction data (X, y)
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary containing train and test data
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Split target data into train and test
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target, y_target, test_size=test_size, random_state=self.random_state
        )
        
        return {
            'source': (X_source, y_source),
            'target_train': (X_target_train, y_target_train),
            'target_test': (X_target_test, y_target_test)
        }
    
    def train_source_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the initial model on source direction data.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            
        Returns:
            Dict[str, Any]: Training results
        """
        return self.model_trainer.evaluate_models(X, y)
    
    def fine_tune_model(self, 
                       source_data: Tuple[np.ndarray, np.ndarray],
                       target_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Fine-tune the model on target direction data.
        
        Args:
            source_data (Tuple[np.ndarray, np.ndarray]): Source direction data (X, y)
            target_data (Tuple[np.ndarray, np.ndarray]): Target direction data (X, y)
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for both models
        """
        # Prepare data
        data = self.prepare_transfer_data(source_data, target_data)
        
        # Train on source data
        source_results = self.train_source_model(*data['source'])
        
        # Fine-tune on target data
        target_results = self.model_trainer.evaluate_models(*data['target_train'])
        
        # Evaluate on target test data
        X_target_test, y_target_test = data['target_test']
        
        # Get predictions from both models
        rf_pred = self.model_trainer.rf_model.predict(X_target_test)
        nn_pred = (self.model_trainer.nn_model(torch.FloatTensor(X_target_test).to(self.model_trainer.device)) > 0.5).float().cpu().numpy().flatten()
        
        # Calculate metrics for fine-tuned models
        fine_tuned_metrics = {
            'random_forest': {
                'accuracy': accuracy_score(y_target_test, rf_pred),
                'precision': precision_score(y_target_test, rf_pred, zero_division=0),
                'recall': recall_score(y_target_test, rf_pred, zero_division=0),
                'f1': f1_score(y_target_test, rf_pred, zero_division=0)
            },
            'neural_network': {
                'accuracy': accuracy_score(y_target_test, nn_pred),
                'precision': precision_score(y_target_test, nn_pred, zero_division=0),
                'recall': recall_score(y_target_test, nn_pred, zero_division=0),
                'f1': f1_score(y_target_test, nn_pred, zero_division=0)
            }
        }
        
        return {
            'source_results': source_results,
            'target_results': target_results,
            'fine_tuned_results': fine_tuned_metrics
        } 