import os
import pickle
import numpy as np                          
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional -> concatenated forward+backward hidden state, so 2x width
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),  # Expects 512 (128 * 4) -> Outputs 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # Output shape: (B, T, 2H)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Pool across the time dimension (dim=1)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)

        # Combine average context and peak feature signals
        pooled = torch.cat((avg_pool, max_pool), dim=1) # (B, 4H)
        out = self.dropout(pooled)
        return self.fc(out) # Self.fc input dim becomes hidden_dim * 4
    
    
class LSTM_Model:
    def __init__(
        self,
        model_path: str = "saved_models/bilstm.pt",
        vocab_path: str = "saved_models/vocab.pkl",
        label_encoder_path: str = "saved_models/label_encoder.pkl",
        max_len: int = 40,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        device: str = None
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.label_encoder_path = label_encoder_path
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.vocab = None
        self.label_encoder = None
        self.model = None

    def load_assets(self) -> None:
        if self.model is None:
            # Check for config file to auto-sync architecture parameters if available
            config_path = os.path.join(os.path.dirname(self.model_path), "config.pkl")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "rb") as f:
                        cfg = pickle.load(f)
                        self.max_len = cfg.get("max_len", self.max_len)
                        self.embed_dim = cfg.get("embed_dim", self.embed_dim)
                        self.hidden_dim = cfg.get("hidden_dim", self.hidden_dim)
                except Exception:
                    pass

            # Load vocabulary dictionary
            with open(self.vocab_path, "rb") as f:
                self.vocab = pickle.load(f)

            # Load scikit-learn LabelEncoder instance
            with open(self.label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)

            pad_idx = self.vocab.get("<pad>", 0)
            vocab_size = len(self.vocab)
            num_classes = len(self.label_encoder.classes_)

            # Instantiate network architecture
            self.model = BiLSTMClassifier(
                vocab_size=vocab_size,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                num_classes=num_classes,
                pad_idx=pad_idx
            ).to(self.device)

            # Load saved state dict weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def _encode_and_pad(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes, indexes, and pads input text."""
        tokens = word_tokenize(text.lower())[:self.max_len]
        length = max(len(tokens), 1)

        unk_idx = self.vocab.get("<unk>", 1)
        pad_idx = self.vocab.get("<pad>", 0)

        ids = [self.vocab.get(token, unk_idx) for token in tokens]
        if len(ids) < self.max_len:
            ids += [pad_idx] * (self.max_len - len(ids))

        x_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        len_tensor = torch.tensor([length], dtype=torch.long)
        return x_tensor, len_tensor

    def predict(self, text: str) -> dict:
        
        self.load_assets()

        if not text or not text.strip():
            return {
                "label": "Invalid Input",
                "confidence": 0.0,
                "probabilities": {}
            }

        x_tensor, len_tensor = self._encode_and_pad(text)

        with torch.no_grad():
            logits = self.model(x_tensor, len_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))

        predicted_label = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            "label": predicted_label,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                cls_name: float(p)
                for cls_name, p in zip(self.label_encoder.classes_, probs)
            }
        }