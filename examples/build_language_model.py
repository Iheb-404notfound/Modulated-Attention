"""
Train a language model on WikiText-2 using BMA.

This script demonstrates how to train a transformer language model with
bilinearly modulated attention and compare against baselines.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse
import math
from pathlib import Path
from tqdm import tqdm

from bma.pytorch import TransformerLM


class LanguageModelingDataset(Dataset):
    """Dataset for causal language modeling."""
    
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        
        x = chunk[:-1]  # input
        y = chunk[1:]   # target
        return x, y


def train_tokenizer(dataset, vocab_size=16000):
    """Train a BPE tokenizer on the dataset."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset["train"]), batch_size):
            yield dataset["train"][i:i + batch_size]["text"]
    
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    return tokenizer


def encode_split(split, tokenizer):
    """Encode a dataset split into token IDs."""
    all_ids = []
    for text in split["text"]:
        if text.strip() == "":
            continue
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
    return all_ids


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model and return perplexity."""
    model.eval()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return math.exp(avg_loss)


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Train or load tokenizer
    tokenizer_path = Path("tokenizer.json")
    if tokenizer_path.exists() and not args.retrain_tokenizer:
        print("Loading existing tokenizer...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print("Training tokenizer...")
        tokenizer = train_tokenizer(dataset, args.vocab_size)
        tokenizer.save(str(tokenizer_path))
    
    # Encode datasets
    print("Encoding datasets...")
    train_ids = encode_split(dataset["train"], tokenizer)
    val_ids = encode_split(dataset["validation"], tokenizer)
    test_ids = encode_split(dataset["test"], tokenizer)
    
    # Create data loaders
    train_dataset = LanguageModelingDataset(train_ids, args.seq_len)
    val_dataset = LanguageModelingDataset(val_ids, args.seq_len)
    test_dataset = LanguageModelingDataset(test_ids, args.seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    print(f"Creating {args.attention_type} model...")
    model = TransformerLM(
        vocab_size=len(tokenizer.get_vocab()),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.seq_len,
        dropout=args.dropout,
        attention_type=args.attention_type
    ).to(device)
    
    n_params = model.count_parameters()
    print(f"Model has {n_params:,} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_ppl = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_ppl = evaluate(model, val_loader, device)
        
        print(f"Train loss: {train_loss:.3f} | Val PPL: {val_ppl:.2f}")
        
        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
            }, f"best_model_{args.attention_type}.pt")
            print(f"Saved best model (PPL: {val_ppl:.2f})")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_ppl = evaluate(model, test_loader, device)
    print(f"Test PPL: {test_ppl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--attention_type", type=str, default="bma",
                        choices=["bma", "standard", "gated"])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Data arguments
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--retrain_tokenizer", action="store_true")
    
    args = parser.parse_args()
    main(args)
