import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from UNetForward import UNet
from NuScenesBevDataset import NuScenesBevDataset
from torch.utils.data import Subset
import os
from tqdm import tqdm

# 1. Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_dataset = NuScenesBevDataset('./processed_data/input_bev', './processed_data/ground_truth')
TRAIN_END_IDX = 350 #ENUMERATED NOT 0 INDEXED
TOTAL_PROCESSED = 404#ENUMERATED NOT 0 INDEXED

train_indices = list(range(0, TRAIN_END_IDX))
val_indices = list(range(TRAIN_END_IDX, TOTAL_PROCESSED))

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# Batch size 4 is safe for Mac memory; shuffle=True is vital for learning
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

# 2. Initialize
model = UNet(n_channels=3, n_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Weighted Loss: Tell the model that missing a car (1) is 5x worse 
# than missing empty road (0). This stops the model from just predicting "all black".
weights = torch.tensor([1.0, 5.0, 10.0, 5.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)



# 3. Training Loop
def train_model(epochs=3):
    checkpoint_path = "best_loss.txt"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            best_v_loss = float(f.read())
            print(f"Resuming training. Current record to beat: {best_v_loss:.4f}")
    else:
        best_v_loss = float('inf')
    
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{epochs} [Train]")

        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()      # Clear old gradients
            output = model(data)       # Forward pass
            loss = criterion(output, target) # Calculate error
            loss.backward()            # Backpropagation (Chain Rule)
            optimizer.step()           # Update weights
            
            current_loss = loss.item()
            total_train_loss += current_loss
            train_pbar.set_postfix(loss=f"{current_loss:.4f}")
            
        #print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    
    #validation phase(per epoch)

        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]" , leave = False)

        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                v_loss = criterion(val_output, val_target)

                total_val_loss += v_loss.item()
                val_pbar.set_postfix(v_loss=f"{v_loss.item():.4f}")
            
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        
        # Save the learned weights if better than the previous one
        if avg_val < best_v_loss:
            best_v_loss = avg_val
            
            # Save the Weights
            torch.save(model.state_dict(), "unet_v1_weights.pth")
            
            # Save the Loss Record to a text file
            with open(checkpoint_path, "w") as f:
                f.write(str(best_v_loss))
                
            print(f"--> New Best Model Saved! (Val Loss: {best_v_loss:.4f})")

if __name__ == "__main__":
    train_model(epochs=15)