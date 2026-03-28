import torch
import numpy as np
from pathlib import Path
from UNetForward import UNet
from NuScenesBevDataset import NuScenesBevDataset
import check_my_tensors as check  # This uses your existing visualization logic
import os

# --- CONFIGURATION ---
MODEL_WEIGHTS = "unet_v1_weights.pth"
INPUT_DIR = './processed_data/input_bev'
GT_DIR = './processed_data/ground_truth'

# Auto-detect dataset size to support any number of files
from NuScenesBevDataset import NuScenesBevDataset as _TempDataset
_temp_dataset = _TempDataset(INPUT_DIR, GT_DIR)
TOTAL_SAMPLES = len(_temp_dataset)
del _temp_dataset

# Set these to choose which samples to test
# Enumerated
START_IDX = 0
END_IDX = TOTAL_SAMPLES  # Test all files

def calculate_iou(preds, labels, n_classes=4):
    """
    Calculates IoU for each class.
    preds, labels: (Height, Width) arrays of class IDs
    """
    ious = []
    # We flatten the arrays to make comparison easier
    preds = preds.flatten()
    labels = labels.flatten()

    for cls in range(n_classes):
        intersection = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or(preds == cls, labels == cls).sum()
        
        if union == 0:
            # If the class isn't present in ground truth or prediction, skip it
            ious.append(float('nan')) 
        else:
            ious.append(intersection / union)
            
    return ious

def run_test():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    # This automatically finds all .npy files you just preprocessed
    dataset = NuScenesBevDataset(INPUT_DIR, GT_DIR)
    
    if len(dataset) < END_IDX:
        print(f"Warning: Dataset only has {len(dataset)} samples. Adjusting END_IDX.")
        actual_end = len(dataset)
    else:
        actual_end = END_IDX

    # 3. Initialize and Load Model
    model = UNet(n_channels=3, n_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("Successfully loaded trained weights.")
    except FileNotFoundError:
        print("Error: .pth file not found. Did you run train_model() yet?")
        return

    model.eval() # CRITICAL: Sets model to inference mode

    # 4. Loop through the specified range
    cumulative_loss = 0
    all_ious = []
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 10.0, 5.0]).to(device))

    print(f"Calculating Cumulative Loss for indices {START_IDX} to {END_IDX-1}...")
    
    with torch.no_grad():
        for i in range(START_IDX, END_IDX):
            x_tensor, y_tensor = dataset[i]
            x_tensor = x_tensor.unsqueeze(0).to(device)
            y_tensor = y_tensor.unsqueeze(0).to(device)
            
            output = model(x_tensor)
            loss = criterion(output, y_tensor)
            cumulative_loss += loss.item()

            pred_ids = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            gt_ids = y_tensor.cpu().numpy()

            current_iou = calculate_iou(pred_ids, gt_ids)
            all_ious.append(current_iou)

            # Optional: Keep visualization for ONLY the first test sample to spot-check
            if i == START_IDX:
                pred_ids = torch.argmax(output, dim=1)
                token = dataset.input_files[i].name.replace("_x.npy", "")
                check.visualize_processed_data(pred_ids, token)
            
    mean_per_class = np.nanmean(all_ious,axis=0)
    overall_miou = np.nanmean(mean_per_class)
    
    avg_test_loss = cumulative_loss / (END_IDX - START_IDX)
    print(f"\n--- FINAL TEST RESULTS ---")
    print(f"Average Cumulative Loss over 50 samples: {avg_test_loss:.4f}")
    

    checkpoint_path = "best_loss.txt"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            best_v_loss = float(f.read())
            print(f"Recorded Best Value loss: {best_v_loss:.4f}")
    else:
        best_v_loss = float('inf')
    print(f"best value loss------>{best_v_loss}\n")

    print("\n--- mIoU TEST RESULTS ---")
    classes = ["Background", "Vehicle", "Pedestrian", "Obstacle"]
    for idx, name in enumerate(classes):
        print(f"{name} IoU: {mean_per_class[idx]:.4f}")
    
    print(f"-------------------------")
    print(f"Final mIoU Score: {overall_miou:.4f}")

if __name__ == "__main__":
    run_test()