# Run this script with sentence_transformers 2.2.2 and torch 1.13
# torch 2 is producing a different model format
# the output must contain pytorch_model.bin 

import os
from sentence_transformers import SentenceTransformer

def download_model():
    """Download and save the all-MiniLM-L6-v2 model locally"""
    
    model_path = "./model"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    print("Downloading all-MiniLM-L6-v2 model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    
    # Verify the model was saved correctly
    model_files = os.listdir(model_path)
    print(f"Model files saved: {model_files}")
    
    if 'pytorch_model.bin' in model_files:
        print("✓ Model saved successfully with pytorch_model.bin format")
    else:
        print("⚠ Warning: pytorch_model.bin not found in saved model")
    
    print("Model download and save completed!")

if __name__ == "__main__":
    download_model()
