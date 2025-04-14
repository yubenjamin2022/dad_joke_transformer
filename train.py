import pandas as pd
import argparse
from tqdm import tqdm
from model import Transformer, TextCSVDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer
import os


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--model_dim", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--depth", type=float, default=8, help="Learning rate")
    parser.add_argument("--num_heads", type=float, default=8, help="Learning rate")
    parser.add_argument("--mlp_dim", type=float, default=128, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    return parser.parse_args()

def read_dataset(csv_file, text_col):
    """Read CSV dataset"""

    df = pd.read_csv(csv_file, encoding = 'latin1')
    return df[text_col].values

def training_loop(num_epochs, model, lr, dataloader, tokenizer):
    """Training loop for tranformer model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs, eta_min = 1e-5)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    for epoch in range(1, num_epochs+1):
        with tqdm(dataloader) as pbar:
            for batch, target in pbar:
                optimizer.zero_grad()
                batch, target = batch.to(device), target.to(device)

                batch = model.embed(batch)
                output = model(batch)

                loss = criterion(output.permute(0, 2, 1), target) # loss for backprop
                loss = torch.where(target == tokenizer.pad_token_id).item() # gets the padding tokens and sets the loss to 0

                non_pad_count = torch.sum(target != tokenizer.pad_token_id).item()

                avg_loss = torch.sum(loss).item() / non_pad_count

                loss = loss.mean()

                loss.backward()

                optimizer.step()
                scheduler.step()

                if loss_ema is None:
                    loss_ema = loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * avg_loss

                # Update progress bar with loss and current learning rate
            pbar.set_description(f"| Epoch {epoch+1}/{num_epochs} | Loss: {round(avg_loss, 3)}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    model_path = "model_epoch_" + str(epoch) + ".pt"
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join('Checkpoints', model_path))                   

def main():
    args = parse_args()
    jokes = read_dataset(args.data_dir, text_col = "joke")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add padding token to pad underfilled jokes
    seq_len = 256

    model = Transformer(vocab_size = 50258, dim=args.model_dim, depth = args.depth, heads=args.num_heads, 
                        dim_head=args.model_dim//args.num_heads, mlp_dim=args.mlp_dim, seq_len=seq_len)
    
    dataset = TextCSVDataset(jokes, seq_len, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    training_loop(args.epochs, model, args.lr, dataloader, tokenizer)
    
if __name__ == "__main__":
    main()