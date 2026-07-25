#train_lstm_handover.py

"""
Treina uma LSTM para IMITAR a politica de handover vertical presente no
dataset gerado por vho_sim.py (log_decisions=True) / optimize_vho.py, com o
objetivo de usar essa rede como um esquema adicional de decisao de handover
("LSTM-VHO") dentro de run_experiments.py, para comparacao com I-VHO, D-VHO
e LA-VHO.

DIFERENCA IMPORTANTE em relacao ao script original (previsao de velocidade
do vento): aqui o problema e de CLASSIFICACAO BINARIA, nao regressao. Dado
um historico recente do estado do enlace (cobertura geometrica, bloqueio por
sombreamento, disponibilidade do link, velocidade, posicao e o modo
anterior), a rede preve se o usuario deveria estar conectado via VLC (1) ou
via WLAN/RF (0) no passo atual. Por isso a loss e BCEWithLogitsLoss (nao
MSE) e as metricas de avaliacao sao de classificacao (acuracia, precisao,
recall, F1, ROC-AUC, matriz de confusao) em vez de RMSE/MAE/R2 -- ainda
assim reporto MAE/RMSE entre a probabilidade prevista e o rotulo (0/1) como
uma medida adicional de "erro", ja que voce pediu esse calculo.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# 0. CONFIGURACAO
# =====================================================

CSV_PATH = "output/handover_decisions_dataset.csv"

WINDOW_SIZE = 20            # passos de historico (DT=0.25s -> 5s de janela)
FEATURE_COLS = ["x", "y", "vel", "link_available", "geo_cov", "shadow_blocked", "mode_before"]
CONTINUOUS_COLS = ["x", "y", "vel"]   # colunas normalizadas com MinMaxScaler
TARGET_COL = "mode_after"             # 1 = VLC, 0 = RF

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
SEED = 42

USE_EARLY_STOPPING = True   # liga/desliga o early stopping
PATIENCE = 7                 # epocas sem melhora na loss de validacao antes de parar
MIN_DELTA = 1e-4             # melhora minima para contar como progresso

MODEL_OUT = "output/lstm_vho_model.pt"
SCALER_OUT = "output/lstm_vho_scaler.pkl"
META_OUT = "output/lstm_vho_meta.json"

torch.manual_seed(SEED)
np.random.seed(SEED)

# =====================================================
# 1. CARREGAR DADOS
# =====================================================

print(f"[1/9] Carregando dataset: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

for col in ["link_available", "geo_cov", "shadow_blocked", "mode_before", TARGET_COL]:
    df[col] = df[col].astype(int)

print(f"    total de linhas: {len(df)} | usuarios unicos: {df['user'].nunique()}")
print(f"    proporcao VLC no dataset completo: {df[TARGET_COL].mean():.3f}")

# =====================================================
# 2. JANELAS POR USUARIO (evita misturar historico de trajetorias diferentes)
# =====================================================

print(f"[2/9] Criando janelas (window_size={WINDOW_SIZE}) por usuario...")

X_list, y_list, user_of_window = [], [], []

for user_id, g in df.groupby("user"):
    g = g.sort_values("step").reset_index(drop=True)
    feats = g[FEATURE_COLS].values.astype(np.float32)
    target = g[TARGET_COL].values.astype(np.float32)

    for i in range(WINDOW_SIZE - 1, len(g)):
        X_list.append(feats[i - WINDOW_SIZE + 1: i + 1])
        y_list.append(target[i])
        user_of_window.append(user_id)

X = np.array(X_list)                      # (N, window, n_features)
y = np.array(y_list)                      # (N,)
user_of_window = np.array(user_of_window)

print(f"    X shape: {X.shape} | y shape: {y.shape} | proporcao VLC(1) nas janelas: {y.mean():.3f}")

# =====================================================
# 3. SPLIT POR USUARIO (70/15/15) -- evita vazamento treino/teste
# =====================================================

print("[3/9] Dividindo em treino/validacao/teste por usuario...")

users = np.sort(df["user"].unique())
rng = np.random.default_rng(SEED)
rng.shuffle(users)

n_users = len(users)
train_end = int(n_users * 0.70)
val_end = int(n_users * 0.85)

train_users = set(users[:train_end])
val_users = set(users[train_end:val_end])
test_users = set(users[val_end:])

train_mask = np.isin(user_of_window, list(train_users))
val_mask = np.isin(user_of_window, list(val_users))
test_mask = np.isin(user_of_window, list(test_users))

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"    Train: {X_train.shape} ({len(train_users)} usuarios)")
print(f"    Val  : {X_val.shape} ({len(val_users)} usuarios)")
print(f"    Test : {X_test.shape} ({len(test_users)} usuarios)")

# =====================================================
# 4. NORMALIZACAO (apenas colunas continuas: x, y, vel)
# =====================================================

print("[4/9] Normalizando features continuas (x, y, vel)...")

cont_idx = [FEATURE_COLS.index(c) for c in CONTINUOUS_COLS]

scaler_X = MinMaxScaler()
scaler_X.fit(X_train[:, :, cont_idx].reshape(-1, len(cont_idx)))


def apply_scaler(Xarr):
    n, w, f = Xarr.shape
    Xc = Xarr.copy()
    flat = Xc[:, :, cont_idx].reshape(-1, len(cont_idx))
    flat = scaler_X.transform(flat)
    Xc[:, :, cont_idx] = flat.reshape(n, w, len(cont_idx))
    return Xc


X_train = apply_scaler(X_train)
X_val = apply_scaler(X_val)
X_test = apply_scaler(X_test)

with open(SCALER_OUT, "wb") as fpkl:
    pickle.dump(scaler_X, fpkl)

# =====================================================
# 5. TENSORES / DATALOADERS
# =====================================================

print("[5/9] Convertendo para tensores PyTorch...")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# 6. MODELO LSTM (classificacao binaria)
# =====================================================

class LSTMHandoverClassifier(nn.Module):
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

        x = x[:, -1, :]   # ultimo timestep

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)   # logit (sem sigmoid -- usamos BCEWithLogitsLoss)

        return x.squeeze(-1)


model = LSTMHandoverClassifier(n_features=len(FEATURE_COLS))
print(model)

# =====================================================
# 7. LOSS, OTIMIZADOR E TREINO
# =====================================================

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
best_state_dict = None
epochs_no_improve = 0
stopped_epoch = None

print(f"[6/9] Treinando por ate {EPOCHS} epocas (batch_size={BATCH_SIZE}, "
      f"early_stopping={'ON' if USE_EARLY_STOPPING else 'OFF'}"
      + (f", patience={PATIENCE})" if USE_EARLY_STOPPING else ")"))

for epoch in range(EPOCHS):

    model.train()
    epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(yb)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        epoch_correct += (preds == yb).sum().item()
        epoch_n += len(yb)

    train_loss = epoch_loss / epoch_n
    train_acc = epoch_correct / epoch_n

    model.eval()
    with torch.no_grad():
        val_loss_total, val_correct, val_n = 0.0, 0, 0
        for xb, yb in val_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            val_loss_total += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_n += len(yb)

        val_loss = val_loss_total / val_n
        val_acc = val_correct / val_n

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"    Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    if USE_EARLY_STOPPING:
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"        (sem melhora na val_loss ha {epochs_no_improve}/{PATIENCE} epocas)")

        if epochs_no_improve >= PATIENCE:
            stopped_epoch = epoch + 1
            print(f"    Early stopping acionado na epoca {stopped_epoch} "
                  f"(melhor val_loss: {best_val_loss:.4f})")
            break

# se o early stopping estava ligado e encontrou pesos melhores, restaura-os
if USE_EARLY_STOPPING and best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    print(f"    Restaurando pesos da melhor epoca (val_loss={best_val_loss:.4f})")

# =====================================================
# 8. GRAFICOS DE TREINO (para o artigo)
# =====================================================

print("[7/9] Gerando graficos de treino...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(train_losses, label="Treino")
axes[0].plot(val_losses, label="Validacao")
if USE_EARLY_STOPPING and stopped_epoch is not None:
    axes[0].axvline(stopped_epoch - 1, color="gray", linestyle="--",
                     label=f"Early stop (epoca {stopped_epoch})")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Curva de perda (Loss) - LSTM-VHO")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(train_accs, label="Treino")
axes[1].plot(val_accs, label="Validacao")
if USE_EARLY_STOPPING and stopped_epoch is not None:
    axes[1].axvline(stopped_epoch - 1, color="gray", linestyle="--",
                     label=f"Early stop (epoca {stopped_epoch})")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("Acuracia")
axes[1].set_title("Curva de acuracia - LSTM-VHO")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Treinamento da LSTM-VHO")
fig.tight_layout()
fig.savefig("output/lstm_training_curves.png", dpi=150)
plt.close(fig)

# =====================================================
# 9. AVALIACAO NO TESTE + GRAFICOS DE ERRO
# =====================================================

print("[8/9] Avaliando no conjunto de teste...")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.sigmoid(test_logits).numpy()

y_true = y_test
y_prob = test_probs
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
n_classes_test = len(np.unique(y_true))
if n_classes_test < 2:
    print(f"\nAVISO: o conjunto de teste so contem uma classe ({np.unique(y_true)}). "
          f"ROC-AUC e Average Precision nao sao definidos nesse caso — "
          f"considere aumentar n_iter no dataset ou revisar o split treino/val/teste.")
    auc = float("nan")
    ap = float("nan")
else:
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

mae = mean_absolute_error(y_true, y_prob)     # erro entre prob. prevista e rotulo 0/1
rmse = np.sqrt(mean_squared_error(y_true, y_prob))
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("\n===== METRICAS NO TESTE (LSTM-VHO) =====")
print(f"Acuracia           : {acc:.4f}")
print(f"Precisao           : {prec:.4f}")
print(f"Recall             : {rec:.4f}")
print(f"F1-score           : {f1:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print(f"Average Precision  : {ap:.4f}")
print(f"MAE (prob vs alvo) : {mae:.4f}")
print(f"RMSE (prob vs alvo): {rmse:.4f}")
print("Matriz de confusao (linhas=real, colunas=predito):\n", cm)

# --- grafico: matriz de confusao ---
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["RF (0)", "VLC (1)"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["RF (0)", "VLC (1)"])
ax.set_xlabel("Predito"); ax.set_ylabel("Real")
ax.set_title("Matriz de confusao - LSTM-VHO (teste)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("output/lstm_confusion_matrix.png", dpi=150)
plt.close(fig)

# --- grafico: curva ROC ---
if n_classes_test < 2:
    print("Pulando graficos de ROC e Precisao-Recall (teste com uma unica classe).")
else:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Aleatorio")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curva ROC - LSTM-VHO (teste)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/lstm_roc_curve.png", dpi=150)
    plt.close(fig)

    # --- grafico: curva Precisao-Recall ---
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec_curve, prec_curve, label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisao")
    ax.set_title("Curva Precisao-Recall - LSTM-VHO (teste)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/lstm_pr_curve.png", dpi=150)
    plt.close(fig)

# =====================================================
# 10. SALVAR MODELO + METADADOS (para uso no run_experiments.py)
# =====================================================

print("[9/9] Salvando modelo e metadados...")

torch.save(model.state_dict(), MODEL_OUT)

meta = {
    "feature_cols": FEATURE_COLS,
    "continuous_cols": CONTINUOUS_COLS,
    "window_size": WINDOW_SIZE,
    "n_features": len(FEATURE_COLS),
    "early_stopping": {
        "enabled": USE_EARLY_STOPPING,
        "patience": PATIENCE,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val_loss if USE_EARLY_STOPPING else None,
        "epochs_trained": len(train_losses),
    },
    "test_metrics": {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": auc, "average_precision": ap, "mae": mae, "rmse": rmse,
    },
}
with open(META_OUT, "w") as fjson:
    json.dump(meta, fjson, indent=2)

print("\nOK - modelo salvo em", MODEL_OUT)
print("Graficos salvos em output/lstm_training_curves.png, lstm_confusion_matrix.png, "
      "lstm_roc_curve.png, lstm_pr_curve.png")