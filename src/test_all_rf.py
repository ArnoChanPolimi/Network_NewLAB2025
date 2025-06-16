import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_feature_matrix
from src.model_training import ModelTrainer

def process_all_datasets():
    """Process all dataset files and combine them for training."""
    # Get the dataset directory
    dataset_dir = Path("dataset")
    
    # Initialize lists to store data
    all_features = []
    all_labels = []
    
    # Process each capture directory
    for capture_dir in ["1st_capture", "2nd_capture"]:
        capture_path = dataset_dir / capture_dir
        if not capture_path.exists():
            print(f"Warning: {capture_path} does not exist")
            continue
            
        # Process each CSV file in the directory
        for csv_file in capture_path.glob("*.csv"):
            print(f"\nProcessing {csv_file.name}...")
            
            try:
                # Load and clean data
                df = load_data(csv_file)
                df_cleaned = clean_data(df)
                
                # Create features
                features_dict = create_feature_matrix(df_cleaned['delay_ms'].values)
                
                # Append features and labels
                all_features.append(features_dict['feature_matrix'])
                all_labels.append(features_dict['labels'])
                
                # Print statistics for this file
                print(f"\nFile Statistics:")
                print(f"Total samples: {len(features_dict['labels'])}")
                print(f"Positive samples: {sum(features_dict['labels'])}")
                print(f"Positive ratio: {sum(features_dict['labels'])/len(features_dict['labels'])*100:.2f}%")
                
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                continue
    
    # Combine all features and labels
    if all_features and all_labels:
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print("\nCombined Dataset Statistics:")
        print(f"Total samples: {len(y)}")
        print(f"Positive samples: {sum(y)}")
        print(f"Positive ratio: {sum(y)/len(y)*100:.2f}%")
        
        # Train and evaluate model
        trainer = ModelTrainer()
        results = trainer.train_random_forest(X, y)
        
        return results
    else:
        print("No valid data found in any files")
        return None

if __name__ == "__main__":
    results = process_all_datasets() 