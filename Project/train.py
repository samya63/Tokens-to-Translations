import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

os.makedirs("checkpoints", exist_ok=True)

from Transformer import Transformer
from dataset import TranslationDataset, collate_fn

torch.manual_seed(42)

#hyperparameters
batch_size=128
epochs=20
learning_rate=3e-4
d_model=256
n_heads=8
n_blocks=6
max_len=256



def save_checkpoint(filename, model, optimizer, epoch):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch+1,
        "hyperparameters": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_blocks": n_blocks,
            "max_len": max_len,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
    }

    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return checkpoint["epoch"], checkpoint["hyperparameters"]

def estimate_loss(loader,model,criterion):
    was_training = model.training
    model.eval()
    total_loss=0
    with torch.no_grad():
        for src,tgt in loader:
            decoder_input = tgt[:,:-1]
            labels=tgt[:,1:]
            logits=model(src,decoder_input)
            B,T,V=logits.shape
            loss=criterion(
                logits.reshape(B*T,V),
                labels.reshape(B*T)
                )
            total_loss+=loss.item()
    if was_training:
        model.train()
    return total_loss/len(loader)

def train():
    print("Starting Training")
    train_dataset= TranslationDataset("train")
    val_dataset= TranslationDataset("validation")

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,drop_last=True)

    model = Transformer(
    d_model=d_model,
    n_head=n_heads,
    max_len=max_len,
    n_blocks=n_blocks,
    src_vocab_size=train_dataset.src_vocab_size,
    tgt_vocab_size=train_dataset.tgt_vocab_size,
    )

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    start_epoch=0
    best_val_loss = float("inf")
    best_model_path="checkpoints/best_model.pt"
    last_model_path="checkpoints/last_model.pt"
    if os.path.exists(best_model_path):
        print(f"Loading {best_model_path} to calculate baseline loss...")
        load_checkpoint(best_model_path,model)
        best_val_loss = estimate_loss(val_loader, model, criterion)
        print(f"True Best Val Loss set to: {best_val_loss:.4f}")

    if os.path.exists(last_model_path):
        print(f"Loading {last_model_path} to resume training...")
        # Note: We load the optimizer here so momentum/learning rates are restored
        start_epoch, _ = load_checkpoint(last_model_path, model, optimizer)
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("No last_model checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch,epochs):
        print(f"Starting Epoch :{epoch+1}/{epochs}")
        model.train()

        running_train_loss = 0.0

        for batch_idx,(src,tgt) in enumerate(train_loader):
            decoder_input = tgt[:,:-1]
            labels=tgt[:,1:]
            logits=model(src,decoder_input)
            B,T,V=logits.shape
            loss=criterion(
                logits.reshape(B*T,V),
                labels.reshape(B*T)
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
            if batch_idx % 100==0:
                val_loss = estimate_loss(val_loader,model,criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                    "checkpoints/best_model.pt",
                    model,
                    optimizer,
                    epoch,
                    )
                    print("New best model saved!")



        train_loss = running_train_loss / len(train_loader)
        val_loss = estimate_loss(val_loader,model,criterion)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                "checkpoints/best_model.pt",
                model,
                optimizer,
                epoch,
            )
            print("New best model saved!")

        save_checkpoint(
        "checkpoints/last_model.pt",
        model,
        optimizer,
        epoch,
        )

if __name__ == "__main__":
    train()