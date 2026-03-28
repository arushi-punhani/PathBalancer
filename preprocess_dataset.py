import os
import numpy as np
from tqdm import tqdm # Install this for a nice progress bar: pip install tqdm
from nuscenes.nuscenes import NuScenes
from pathlib import Path


# Import your functions from the other files
# Ensure sampleToBEVgrid.py and groundTruthtoBEVgrid.py are in the same folder
from sampleToBEVgrid import get_bev_input
from groundTruthtoBEVgrid import get_bev_gt

# Find relative path
current_script_path = Path(__file__).resolve()
parent_folder = current_script_path.parent
data_path = parent_folder / "trainingData" / "v1.0-mini"

# Configuration
DATAROOT = data_path
VERSION = 'v1.0-mini'
OUTPUT_DIR_X = './processed_data/input_bev'
OUTPUT_DIR_Y = './processed_data/ground_truth'
NUM_SAMPLES_TO_PROCESS = 563 # desired 'n' -> is now offset by start_0(it starts preprocessing from there->still 0 indexed)
                            # classic python slice logic 

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR_X, exist_ok=True)
os.makedirs(OUTPUT_DIR_Y, exist_ok=True)

# Initialize nuScenes
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

def run_preprocessing(n):
    # nusc.sample is a list of all keyframes
    start_0 = 4
    samples_to_process = nusc.sample[start_0:n]
    
    print(f"Starting preprocessing for {n-start_0} samples...")
    
    for i, sample in enumerate(tqdm(samples_to_process)):
        token = sample['token']
        
        try:
            # 1. Generate the Input Tensor (Rank-3: Height, Intensity, Density)
            x_data = get_bev_input(nusc,token)
            
            # 2. Generate the Ground Truth (Rank-2: Class IDs)
            y_data = get_bev_gt(nusc,token)
            
            # 3. Save as .npy files
            # Using the token as the filename ensures they stay matched
            np.save(os.path.join(OUTPUT_DIR_X, f"{token}_x.npy"), x_data)
            np.save(os.path.join(OUTPUT_DIR_Y, f"{token}_y.npy"), y_data)
            
        except Exception as e:
            print(f"Error processing sample {i} ({token}): {e}")

if __name__ == "__main__":
    run_preprocessing(NUM_SAMPLES_TO_PROCESS)