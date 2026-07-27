"""
Inference wrapper for the LSTM-VHO model trained in `train_lstm_handover.py`. 
It is used by `vho_sim.py` to implement the "LSTM-VHO" scheme: at each simulation step, 
`vho_sim.py` maintains a sliding window of each user's state and queries this policy to decide 
whether the user should be connected via VLC or RF.
"""

import json
import pickle

import numpy as np
import torch
import torch.nn as nn


class LSTMHandoverClassifier(nn.Module):
    """Mesma arquitetura usada em train_lstm_handover.py."""

    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


class LSTMHandoverPolicy:
    """
    Carrega o modelo treinado e os metadados uma unica vez, e oferece um
    metodo `decide` vetorizado que recebe o buffer de janelas de todos os
    usuarios ativos na simulacao (shape: n_users x window_size x n_features,
    na MESMA ordem de colunas de self.feature_cols) e retorna um array
    booleano (True = usar VLC, False = usar RF).
    """

    def __init__(self,
                 model_path="output/lstm_vho_model.pt",
                 scaler_path="output/lstm_vho_scaler.pkl",
                 meta_path="output/lstm_vho_meta.json",
                 threshold=0.5):

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta["feature_cols"]
        self.continuous_cols = self.meta["continuous_cols"]
        self.window_size = self.meta["window_size"]
        self.n_features = self.meta["n_features"]
        self.cont_idx = [self.feature_cols.index(c) for c in self.continuous_cols]
        self.threshold = threshold

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.model = LSTMHandoverClassifier(self.n_features)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def _scale(self, window_batch):
        n, w, f = window_batch.shape
        wb = window_batch.copy()
        flat = wb[:, :, self.cont_idx].reshape(-1, len(self.cont_idx))
        flat = self.scaler.transform(flat)
        wb[:, :, self.cont_idx] = flat.reshape(n, w, len(self.cont_idx))
        return wb

    def decide(self, window_batch):
        """
        window_batch: np.array shape (n_users, window_size, n_features).
        Retorna: np.array booleano shape (n_users,) -> True = usar VLC.
        """
        wb = self._scale(window_batch).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.tensor(wb))
            probs = torch.sigmoid(logits).numpy()
        return probs >= self.threshold
