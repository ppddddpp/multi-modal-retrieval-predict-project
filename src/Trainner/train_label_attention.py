from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from KnowledgeGraph.label_attention import LabelAttention

# Base project dir (3 levels up)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Default model dir
MODEL_DIR = BASE_DIR / "label attention model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_label_attention(
    dataset,
    label_lookup,
    d_emb,
    hidden=64,
    batch_size=32,
    epochs=20,
    lr=1e-3,
    patience=3,
    save_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_to_wandb=False
):
    """
    Train a LabelAttention model.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to train on.
        label_lookup (LabelEmbeddingLookup): The label lookup object.
        d_emb (int): The dimensionality of the embeddings.
        hidden (int, optional): The number of hidden units in the model. Defaults to 64.
        batch_size (int, optional): The batch size. Defaults to 32.
        epochs (int, optional): The number of epochs to train for. Defaults to 20.
        lr (float, optional): The learning rate. Defaults to 1e-3.
        patience (int, optional): The patience for early stopping. Defaults to 3.
        save_path (str, optional): The path to save the model to. Defaults to None, which means the model is saved to MODEL_DIR / "label_attention_model.pt".
        device (str, optional): The device to use. Defaults to "cuda" if available, otherwise "cpu".
        log_to_wandb (bool, optional): Whether to log to Weights and Biases. Defaults to False.

    Returns:
        LabelAttention: The trained model.
    """
    if save_path is None:
        save_path = MODEL_DIR / "label_attention_model.pt"

    model = LabelAttention(d_emb=d_emb, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"LabelAttention Epoch {epoch}/{epochs}"):
            qids, pids, nids = batch
            batch_q, batch_p, batch_n = [], [], []
            mask_q, mask_p, mask_n = [], [], []

            for qid, pid, nid in zip(qids, pids, nids):
                def get_and_mask(rid):
                    emb = label_lookup.get_label_embs(rid)  # [n_labels, d]
                    mask = torch.ones((emb.shape[0],), dtype=torch.bool)
                    return emb, mask
                q_embs, q_mask = get_and_mask(qid)
                p_embs, p_mask = get_and_mask(pid)
                n_embs, n_mask = get_and_mask(nid)
                batch_q.append(q_embs); mask_q.append(q_mask)
                batch_p.append(p_embs); mask_p.append(p_mask)
                batch_n.append(n_embs); mask_n.append(n_mask)

            max_len = max(x.shape[0] for x in batch_q + batch_p + batch_n)

            def pad_stack(tensors, masks):
                emb_out, mask_out = [], []
                for t, m in zip(tensors, masks):
                    pad_len = max_len - t.shape[0]
                    if pad_len > 0:
                        pad_emb = torch.zeros((pad_len, t.shape[1]), device=device)
                        pad_mask = torch.zeros((pad_len,), dtype=torch.bool)
                        t = torch.cat([t, pad_emb], dim=0)
                        m = torch.cat([m, pad_mask], dim=0)
                    emb_out.append(t)
                    mask_out.append(m)
                return torch.stack(emb_out).to(device), torch.stack(mask_out).to(device)

            q_batch, q_mask_batch = pad_stack(batch_q, mask_q)
            p_batch, p_mask_batch = pad_stack(batch_p, mask_p)
            n_batch, n_mask_batch = pad_stack(batch_n, mask_n)

            optimizer.zero_grad()
            q_rep, _ = model(q_batch, mask=q_mask_batch)
            p_rep, _ = model(p_batch, mask=p_mask_batch)
            n_rep, _ = model(n_batch, mask=n_mask_batch)

            loss = triplet_loss(q_rep, p_rep, n_rep)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}  Loss={avg_loss:.4f}")

        if log_to_wandb:
            import wandb
            wandb.log({"la/train_loss": avg_loss, "la/epoch": epoch})

        # early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [*] New best loss. Model saved to {save_path}")
            if log_to_wandb:
                wandb.log({"la/best_loss": best_loss, "la/epoch": epoch})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  [!] Early stopping triggered.")
                if log_to_wandb:
                    wandb.log({"la/early_stop_epoch": epoch})
                break

    # load best model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model
