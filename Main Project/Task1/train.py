import os
import random
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from models.LSTM import BiLSTMClassifier

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


DATASET_PATH = "cellula toxic data (1).csv"
TEXT_COL = "text"
LABEL_COL = "label"
SEED = 42
MAX_LEN_PERCENTILE = 95  
EMBED_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
MIN_FREQ = 2

SAVE_DIR = "models/saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def calculate_dynamic_max_len(texts, percentile: int = 95) -> int:
    """Tokenizes all texts and calculates sequence length at the specified percentile."""
    lengths = [len(word_tokenize(str(text).lower())) for text in texts]
    calculated_max_len = int(np.percentile(lengths, percentile))
    # Ensure at least a minimal sequence length of 1
    return max(calculated_max_len, 1)


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# --- 4. DATASET & DYNAMIC COLLATOR ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len
        self.unk_idx = vocab.get("<unk>", 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower()
        # Truncate sequence to 95th percentile length
        tokens = word_tokenize(text)[: self.max_len]
        
        ids = [self.vocab.get(t, self.unk_idx) for t in tokens]
        if not ids:
            ids = [self.unk_idx]

        return torch.tensor(ids, dtype=torch.long), self.labels[idx]


def dynamic_collate_fn(batch, pad_idx=0):
    """Pads sequences dynamically per batch up to the longest sentence in that batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels


# --- 5. MAIN TRAINING ROUTINE ---
def train():
    # Enforce constant reproducibility
    set_seed(SEED)
    print(f"Using device: {DEVICE} | Global Seed: {SEED}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found!")

    df = pd.read_csv(DATASET_PATH).dropna(subset=[TEXT_COL, LABEL_COL])
    print(f"Dataset loaded: {len(df)} samples.")

    # Calculate MAX_LEN dynamically using 95th percentile
    max_len = calculate_dynamic_max_len(df[TEXT_COL].values, percentile=MAX_LEN_PERCENTILE)
    print(f"Calculated MAX_LEN ({MAX_LEN_PERCENTILE}th percentile): {max_len} tokens")

    # Encode Labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[LABEL_COL])
    num_classes = len(label_encoder.classes_)

    # Build Vocabulary
    vocab = build_vocab(df[TEXT_COL], min_freq=MIN_FREQ)
    pad_idx = vocab["<pad>"]
    print(f"Vocabulary size: {len(vocab)} words.")

    # Stratified Train/Val Split with constant seed
    X_train, X_val, y_train, y_val = train_test_split(
        df[TEXT_COL].values,
        encoded_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=encoded_labels,
    )

    train_ds = TextDataset(X_train, y_train, vocab, max_len)
    val_ds = TextDataset(X_val, y_val, vocab, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_idx),
    )

    # Initialize Network
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        pad_idx=pad_idx,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nStarting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x_batch, len_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_batch, len_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, len_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(x_batch, len_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

    # Export Artifacts
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "bilstm.pt"))
    with open(os.path.join(SAVE_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nTraining Complete! All artifacts exported successfully.")


if __name__ == "__main__":
    train()