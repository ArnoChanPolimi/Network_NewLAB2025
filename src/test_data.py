import pandas as pd
from pathlib import Path
from src.preprocessing import load_data, clean_data

def test_data_loading():
    """Test data loading and preprocessing."""
    # Load a sample file
    file_path = Path('dataset/1st_capture/cpe_a-cpe_b-fiber.csv')
    
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    print("\nFirst few rows of raw data:")
    print(df.head())
    
    print("\nData info:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Clean the data
    print("\nCleaning data...")
    df_clean = clean_data(df)
    
    print("\nFirst few rows of cleaned data:")
    print(df_clean.head())
    
    print("\nPacket loss statistics:")
    total_samples = len(df_clean)
    packet_loss = df_clean['is_packet_loss'].sum()
    print(f"Total samples: {total_samples}")
    print(f"Packet loss events: {packet_loss}")
    print(f"Packet loss rate: {packet_loss/total_samples*100:.2f}%")

if __name__ == "__main__":
    test_data_loading() 