import argparse
import torch
import os
import numpy as np
import time
from torch.utils.data import DataLoader

from model import SlidingDataset, RMSELoss, EarlyStopping, DRAMA
from utils import build_spatial_edge
from deco import ProgressBar
    
def run_training(model: DRAMA, loader, optimizer, loss_fn, device, config, model_path):
    early_stopper = EarlyStopping(patience=config.patience, min_delta=0)
    best_loss = 1e6
    total_time = 0

    for epoch in range(config.EPOCHS):
        model.train()
        model.reset_mem()
        total_loss = 0
        num_batches = 0
        bar = ProgressBar(loader, total=len(loader), unit='batch', desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        start = time.time()
        for batch in bar:
            x = batch.to(device) #[b,c,w]
            
            optimizer.zero_grad()

            x_hat = model(x) #[b,c,w] 
            l_rec = loss_fn(x_hat, x) #scalar

            loss = l_rec
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            avg_loss = total_loss / num_batches

            if num_batches % 10 == 0:
                bar.set_postfix({'Loss': f"{avg_loss:.6f}"})

        end = time.time()
        epoch_time = end - start
        total_time += epoch_time
        remaining_epochs = config.EPOCHS - (epoch + 1)
        eta_seconds = remaining_epochs * (total_time / (epoch + 1))
        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)

        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.6f} -- ETA: {eta_min}m {eta_sec}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"[Epoch {epoch+1}] New best model saved on {model_path} ({avg_loss:.4f})")

        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print(f"[Epoch {epoch+1}] Early stopping at epoch {epoch+1}")
            break

    print(f"⏱️ Average training time per epoch: {total_time / (epoch+1):.2f}s")
    print(f"⏱️ Total training time: {int(total_time//60)}min {int(total_time%60)}s")
    print("Training done.")
    return model

def train(config):
    # -- Prepare data --
    print("-"*100)
    print(f"Training '{config.dataset}-{config.tag}'")
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print(f"Loading dataset...")
    
    # --------------------------------------------------------------------------------------
    # change here for dataset loading
    x_train = np.load(f'./data/{config.dataset}/preprocessed/train.npy') #(T,C)
    # --------------------------------------------------------------------------------------
    
    print("Train:", x_train.shape)
    train_dataset = SlidingDataset(x_train, window_size=config.W, stride=config.S)
    train_loader = DataLoader(train_dataset, batch_size=config.B, shuffle=False, pin_memory=True, drop_last=True)

    # -- Prepare model --
    print("Initializing model...")
    c = x_train.shape[1]
    edge = build_spatial_edge(mode=config.dataset)
        
    model = DRAMA(
        b=config.B,
        d=config.D,
        c=c,
        m=config.M,
        w=config.W,
        h=config.H,
        decay=config.decay, E_spatial=edge, topk=config.K,
        device=device
    ).to(device)

    model_path = (
        f"models/{config.dataset}/"
        f"{config.tag}_W{config.W}_D{config.D}_M{config.M}_H{config.H}_K{config.K}_decay{config.decay:.3f}_B{config.B}.pth"
    )

    print("Model will be saved as", model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if config.pretrained and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Pretrained model loaded from {model_path}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99))
    loss_fn = RMSELoss()
    print(f"Training started for '{model_path}'")
    print("-"*100)
    run_training(model, train_loader, optimizer, loss_fn, device, config, model_path)

def trainNSL(config):
    # -- Prepare data --
    print("-"*100)
    print(f"Training '{config.dataset}-{config.tag}'")
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nsl_projs = list(set([x.split("_")[0] for x in os.listdir(f'./data/{config.dataset}/') if '_train.npy' in x]))

    for it, proj in enumerate(nsl_projs):
        if config.dataset == 'smap' and proj == 'P-2':
            continue

        print("-"*100)
        print(f"[{it}/{len(nsl_projs)}] '{config.dataset}-{proj}-{config.tag}'")
    
        print(f"Loading dataset...")
        # --------------------------------------------------------------------------------------
        # change here for dataset loading
        x_train = np.load(f'./data/{config.dataset}/{proj}_train.npy') #(T,C)
        # --------------------------------------------------------------------------------------
        
        train_dataset = SlidingDataset(x_train, window_size=config.W)
        train_loader = DataLoader(train_dataset, batch_size=config.B, shuffle=False, pin_memory=True, drop_last=True)

        # -- Prepare model 
        print("Initializing model...")
        c = x_train.shape[1]
        edge = build_spatial_edge(mode=config.dataset)        
        model = DRAMA(
            b=config.B,
            d=config.D,
            c=c,
            m=config.M,
            w=config.W,
            h=config.H,
            decay=config.decay, E_spatial=edge, topk=config.K,
            device=device
        ).to(device)

        model_path = (
            f"models/{config.dataset}/"
            f"{proj}_{config.tag}_W{config.W}_D{config.D}_M{config.M}_H{config.H}_K{config.K}_decay{config.decay:.3f}_B{config.B}.pth"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if config.pretrained and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Pretrained model loaded from {model_path}...")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = RMSELoss()
        print(f"Training started for '{model_path}'")
        print("-"*100)
        run_training(model, train_loader, optimizer, loss_fn, device, config, model_path)



if __name__ == "__main__":
    # ---------------------- Config ----------------------
    parser = argparse.ArgumentParser(description="Train anomaly detection model.")
    parser.add_argument("--dataset", type=str, required=True, choices=['swat', 'wadi', 'smap', 'msl'])
    parser.add_argument("--tag", type=str, required=True)

    parser.add_argument("--W", help='window size', type=int, default=5)
    parser.add_argument("--D", help='memory horizon', type=int, default=30)            
    parser.add_argument("--B", help='batch size', type=int, default=64)        
    parser.add_argument("--K", help='topk', type=int, default=5)             
    parser.add_argument("--M", help='embedding dimension', type=int, default=30)          
    parser.add_argument("--H", help='hidden dimension', type=int, default=32)            
    parser.add_argument("--decay", help='memory decay factor', type=float, default=0.05) 
    
    parser.add_argument("--EPOCHS", type=int, default=50)       
    parser.add_argument("--patience", type=int, default=10)     
    parser.add_argument("--lr", type=float, default=0.001)      
    parser.add_argument("--pretrained", action='store_true', default=False)
    config = parser.parse_args()

    if config.dataset == 'wadi':
        config.W = 30 # window size
        config.S = 1 # stride
        config.D = 30 # memory horizon
        config.B = 64 # batch size ; D*B = 1920s = 32 min
        config.M = 30 # embedding dimension
        config.H = 32 # hidden dimension for encoder, decoder
        config.decay = 0.05 # temporal attention weight
        config.K = 5 # topk temporal attention
        
    elif config.dataset == 'swat':
        config.W = 5
        config.S = 1
        config.D = 50
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05

    elif config.dataset == 'smap':
        config.W = 30
        config.S = 1
        config.D = 30
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05

    elif config.dataset == 'msl':
        config.W = 30
        config.S = 1
        config.D = 30
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05

    if config.dataset in ['swat', 'wadi']:
        train(config)
    else:
        trainNSL(config)
