# train_bc.py
# Point d'entre pour l'entrainement en behavioral cloning
# J'ai ptete tout cassé pck au début j'avais réussi a entrainé mais j'avais oublié d'entrainer le critic
# j'ai demandé a gemini de me refaire tout mais il a ptete tout cassé
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import gymnasium as gym
from ray.rllib.core.columns import Columns
import time
import os

from model import PokeTransformerModule

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "runs/pokeformer_bc_" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir) 
    print(f"--- Starting Dual-Head Behavioral Cloning on {device} ---")
    print(f"Logging to: {log_dir}")

    print("Loading Massive Dataset into RAM...")
    data = np.load("expert_data_combined.npz")
    obs_tensor = torch.tensor(data['obs'], dtype=torch.float32)
    action_tensor = torch.tensor(data['actions'], dtype=torch.long)
    # THE FIX: Load the Critic Targets
    value_tensor = torch.tensor(data['values'], dtype=torch.float32)
    
    dataset = TensorDataset(obs_tensor, action_tensor, value_tensor)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 2048 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    obs_space = gym.spaces.Dict({
        "observations": gym.spaces.Box(low=-10.0, high=np.inf, shape=(13, 200), dtype=np.float32),
        "action_mask": gym.spaces.Box(0.0, 1.0, shape=(26,), dtype=np.float32),
    })
    act_space = gym.spaces.Discrete(26)

    model = PokeTransformerModule(
        observation_space=obs_space, action_space=act_space,
        inference_only=False, model_config={}, catalog_class=None,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # THE FIX: Two separate loss functions!
    criterion_actor = nn.CrossEntropyLoss()
    criterion_critic = nn.MSELoss()
    
    checkpoint_path = "bc_checkpoint.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")

    epochs = 15
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_pi_loss = 0.0
        total_v_loss = 0.0
        
        # THE FIX: Unpack all 3 variables
        for batch_idx, (b_obs, b_acts, b_vals) in enumerate(train_loader):
            b_obs, b_acts, b_vals = b_obs.to(device), b_acts.to(device), b_vals.to(device)
            B = b_obs.shape[0]

            batch_dict = {
                Columns.OBS: {
                    "observations": b_obs,
                    "action_mask": torch.ones((B, 26), device=device)
                }
            }

            out = model._forward(batch_dict)
            logits = out[Columns.ACTION_DIST_INPUTS]
            embeddings = out[Columns.EMBEDDINGS]
            
            # Extract Critic prediction from the 2nd token
            v_preds = model.v_head(embeddings[:, 1, :]).squeeze(-1)

            # Calculate Dual Loss
            loss_pi = criterion_actor(logits, b_acts)
            loss_v = criterion_critic(v_preds, b_vals)
            
            # Combine them (Actor is 1.0, Critic is 0.5)
            loss = loss_pi + (0.5 * loss_v)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_pi_loss += loss_pi.item()
            total_v_loss += loss_v.item()

            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Loss/Actor", loss_pi.item(), step)
                writer.add_scalar("Loss/Critic", loss_v.item(), step)
                if batch_idx % 500 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Actor Loss: {loss_pi.item():.4f} | Critic Loss: {loss_v.item():.4f}")

        # Validation Phase
        model.eval()
        val_v_loss = 0.0
        correct = 0
        with torch.no_grad():
            for b_obs, b_acts, b_vals in val_loader:
                b_obs, b_acts, b_vals = b_obs.to(device), b_acts.to(device), b_vals.to(device)
                batch_dict = {
                    Columns.OBS: {
                        "observations": b_obs,
                        "action_mask": torch.ones((b_obs.shape[0], 26), device=device)
                    }
                }
                out = model._forward(batch_dict)
                logits = out[Columns.ACTION_DIST_INPUTS]
                v_preds = model.v_head(out[Columns.EMBEDDINGS][:, 1, :]).squeeze(-1)
                
                val_v_loss += criterion_critic(v_preds, b_vals).item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == b_acts).sum().item()

        avg_train_pi_loss = total_pi_loss / len(train_loader)
        avg_train_v_loss = total_v_loss / len(train_loader)
        avg_val_v_loss = val_v_loss / len(val_loader)
        val_accuracy = correct / val_size
        
        writer.add_scalar("Loss/Train_Epoch_Actor", avg_train_pi_loss, epoch)
        writer.add_scalar("Loss/Train_Epoch_Critic", avg_train_v_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch_Critic", avg_val_v_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)

        print(f"\n=== EPOCH {epoch+1} FINISHED ===")
        print(f"Actor Loss: {avg_train_pi_loss:.4f} | Critic Train Loss: {avg_train_v_loss:.4f} | Critic Val Loss: {avg_val_v_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_pi_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}\n")

    torch.save(model.state_dict(), "pretrained_pokeformer_final.pt")
    writer.close()
    print("Dual-Head Training Complete! Ready for stable RL!")

if __name__ == "__main__":
    main()