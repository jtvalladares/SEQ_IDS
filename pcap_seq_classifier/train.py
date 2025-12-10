"""Loop de entrenamiento y evaluación.

Funciones expuestas:
- train_loop(...)
- evaluate(...)
- save_checkpoint(...)
"""
from typing import Any, Dict
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def train_loop(train_loader,
               val_loader,
               model,
               optimizer,
               scheduler,
               cfg: Dict[str, Any]):

    """
    Implementa loop de entrenamiento con:
    - early stopping
    - checkpoints
    - LR scheduler opcional

    cfg debe contener:
        epochs: int
        device: "cpu" | "cuda"
        early_stop_patience: int
        checkpoint_path: str
    """

    device = cfg.get("device", "cpu")
    epochs = cfg.get("epochs", 10)
    patience = cfg.get("early_stop_patience", 5)
    ckpt_path = cfg.get("checkpoint_path", "checkpoint.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, lengths, labels = batch
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)

            # Packed sequence forward
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            logits = model(packed).squeeze(-1)

            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- Evaluación ----
        val_stats = evaluate(val_loader, model)
        val_loss = val_stats["loss"]

        # Scheduler opcional
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {val_stats['accuracy']:.4f} | F1: {val_stats['f1']:.4f}"
        )

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "cfg": cfg,
                },
                ckpt_path,
            )

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping activado.")
                break

    print("Entrenamiento terminado.")
    print(f"Mejor valid loss: {best_val_loss:.4f} en epoch {epoch - patience_counter}")


def evaluate(loader, model) -> Dict[str, float]:
    """Evalúa el modelo en un loader completo.

    Retorna métricas:
    - loss
    - accuracy
    - precision
    - recall
    - f1
    - auc
    """

    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            x, lengths, labels = batch
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            logits = model(packed).squeeze(-1)

            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_logits.extend(probs.cpu().numpy().tolist())
            all_preds.extend((probs > 0.5).int().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)

    # Métricas sklearn
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.0  # ocurre si todas las etiquetas son iguales

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Guarda un checkpoint completo."""
    torch.save(state, path)
