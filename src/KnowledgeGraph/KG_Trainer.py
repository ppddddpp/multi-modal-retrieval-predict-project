from pathlib import Path
import json
import csv
import os
import numpy as np
import random
from typing import List, Optional, Tuple, Dict
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KG_DIR = BASE_DIR / "knowledge_graph"
KG_DIR.mkdir(parents=True, exist_ok=True)

class TransEModel(nn.Module):
    def __init__(self, n_nodes: int, n_rels: int, emb_dim: int = 200, p_norm: int = 1):
        super().__init__()
        self.ent = nn.Embedding(n_nodes, emb_dim)
        self.rel = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight.data)
        nn.init.xavier_uniform_(self.rel.weight.data)
        self.p = p_norm

    def score(self, s_idx, r_idx, o_idx):
        e_s = self.ent(s_idx)
        e_o = self.ent(o_idx)
        r = self.rel(r_idx)
        diff = e_s + r - e_o
        if self.p == 1:
            return torch.norm(diff, p=1, dim=1)
        else:
            return torch.norm(diff, p=2, dim=1)

class KGTransETrainer:
    """
    Class wrapper for training TransE on the triples.csv created by KGBuilder.

    Example:
        tr = KGTransETrainer(kg_dir="kg", emb_dim=128)
        tr.load_triples()
        tr.train(epochs=10)
        tr.save_embeddings("kg/node_emb.npy", "kg/rel_emb.npy")
    """

    def __init__(self, kg_dir: str = "kg", emb_dim: int = 200, joint_dim: Optional[int] = None, margin: float = 1.0, 
                    lr: float = 1e-3, curated_factor: float = 3.0, device: Optional[str] = None):
        if kg_dir is None:
            self.kg_dir = KG_DIR
        else:
            self.kg_dir = Path(kg_dir) if Path(kg_dir).is_absolute() else (BASE_DIR / kg_dir)
        self.emb_dim = emb_dim
        self.joint_dim = joint_dim or emb_dim  # default to emb_dim
        self.margin = margin
        self.lr = lr
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.curated_factor = curated_factor
        # these will be filled by load_triples()
        self.node2id = {}
        self.rel2id = {}
        self.triples = []  # tuples (s,r,o,conf,src)
        self.train_triples = []
        self.val_triples = []
        self.model = None
        self.optimizer = None

        if self.joint_dim != self.emb_dim:
            self.proj_to_kg = nn.Linear(self.joint_dim, self.emb_dim, bias=False).to(self.device)
        else:
            self.proj_to_kg = nn.Identity()

    def load_maps(self):
        with (self.kg_dir / "node2id.json").open(encoding='utf8') as f:
            self.node2id = json.load(f)
        with (self.kg_dir / "relation2id.json").open(encoding='utf8') as f:
            self.rel2id = json.load(f)

    def load_triples(self, triples_csv: str = None):
        # default to kg/triples.csv
        tpath = Path(triples_csv) if triples_csv else self.kg_dir / "triples.csv"
        if not tpath.exists():
            raise FileNotFoundError(tpath)
        # ensure maps loaded
        self.load_maps()
        self.triples = []
        with tpath.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = int(row['s_id']); r = int(row['r_id']); o = int(row['o_id'])
                conf = float(row.get('confidence', 1.0))
                src = row.get('source', 'extracted')

                # boost curated edges
                if src == "curated":
                    conf *= getattr(self, "curated_factor", 3.0)
                
                self.triples.append((s, r, o, conf, src))
        print(f"[KGTransETrainer] loaded {len(self.triples)} triples")

        # train/val split
        random.shuffle(self.triples)
        split = int(0.9 * len(self.triples))
        self.train_triples = self.triples[:split]
        self.val_triples = self.triples[split:]

        # build train-only arrays
        self.pos_s = np.array([t[0] for t in self.train_triples], dtype=np.int64)
        self.pos_r = np.array([t[1] for t in self.train_triples], dtype=np.int64)
        self.pos_o = np.array([t[2] for t in self.train_triples], dtype=np.int64)
        self.pos_conf = np.array([t[3] for t in self.train_triples], dtype=np.float32)

        # init model
        n_nodes = len(self.node2id)
        n_rels = len(self.rel2id)
        self.model = TransEModel(n_nodes, n_rels, self.emb_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self,
            epochs: int = 5,
            batch_size: int = 1024,
            normalize: bool = True,
            save_every: int = 1,
            wandb_config: Optional[dict] = None,
            log_to_wandb: bool = True,
            patience: Optional[int] = None,    # NEW: early stopping patience
            metric: str = "mrr"                # NEW: which metric to monitor
            ):
        """
        Train KG TransE embeddings.

        - Supports optional early stopping with patience.
        - Saves best checkpoint (by chosen metric).
        - Robust wandb handling.
        """
        if self.model is None:
            raise RuntimeError("Call load_triples() first.")

        n = len(self.pos_s)
        steps = max(1, (n + batch_size - 1) // batch_size)

        # wandb init bookkeeping
        started_wandb = False
        if log_to_wandb:
            try:
                if getattr(wandb, "run", None) is None:
                    run_name = wandb_config.get("name") if wandb_config and "name" in wandb_config \
                            else f"kg_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    init_kwargs = {
                        "project": wandb_config.get("project", "multi-modal-kg") if wandb_config else "multi-modal-kg",
                        "name": run_name,
                        "config": wandb_config if wandb_config else None,
                        "reinit": True,
                    }
                    if os.environ.get("WANDB_MODE") == "offline" or (wandb_config and wandb_config.get("mode") == "offline"):
                        init_kwargs["mode"] = "offline"
                    wandb.init(**init_kwargs)
                    started_wandb = True
            except Exception as e:
                print(f"[WARN] wandb.init() failed — disabling wandb logging: {e}")
                log_to_wandb = False

        # early stopping bookkeeping
        best_val = -float("inf")
        bad_epochs = 0

        for epoch in tqdm(range(1, epochs + 1), desc="KG Train", unit="epoch"):
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            epoch_loss = 0.0

            for step in tqdm(range(steps), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False):
                start = step * batch_size
                end = min((step + 1) * batch_size, n)
                if start >= end:
                    continue
                batch_idx = idxs[start:end]

                s_batch = torch.LongTensor(self.pos_s[batch_idx]).to(self.device)
                r_batch = torch.LongTensor(self.pos_r[batch_idx]).to(self.device)
                o_batch = torch.LongTensor(self.pos_o[batch_idx]).to(self.device)
                conf_batch = torch.FloatTensor(self.pos_conf[batch_idx]).to(self.device)

                # positive scores
                pos_scores = self.model.score(s_batch, r_batch, o_batch)

                # negative sampling
                corrupt_head = np.random.rand(len(batch_idx)) < 0.5
                neg_s = self.pos_s[batch_idx].copy()
                neg_o = self.pos_o[batch_idx].copy()
                rand_nodes = np.random.randint(0, self.model.ent.num_embeddings, size=len(batch_idx))
                neg_s[corrupt_head] = rand_nodes[corrupt_head]
                neg_o[~corrupt_head] = rand_nodes[~corrupt_head]
                neg_s_t = torch.LongTensor(neg_s).to(self.device)
                neg_o_t = torch.LongTensor(neg_o).to(self.device)

                neg_scores = self.model.score(neg_s_t, r_batch, neg_o_t)

                # margin ranking loss
                loss_sample = torch.relu(pos_scores + self.margin - neg_scores)
                loss = (loss_sample * conf_batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())

            # normalize embeddings
            if normalize:
                with torch.no_grad():
                    w = self.model.ent.weight.data
                    denom = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
                    self.model.ent.weight.data = w / denom

            avg_loss = epoch_loss / max(1, steps)
            print(f"[KGTransETrainer] Epoch {epoch}/{epochs} avg_loss={avg_loss:.6f}")

            # prepare logging payload
            log_data = {"kg/epoch": epoch, "kg/loss": avg_loss}

            # evaluate
            if hasattr(self, "val_triples") and len(self.val_triples) > 0:
                try:
                    mrr, hits1, hits10 = self.evaluate(self.val_triples, k=10)
                    print(f"[Eval] MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@10={hits10:.4f}")
                    log_data.update({
                        "kg/val_mrr": mrr,
                        "kg/val_hits1": hits1,
                        "kg/val_hits10": hits10,
                    })

                    # early stopping
                    val_score = {"mrr": mrr, "hits1": hits1, "hits10": hits10}[metric]
                    if val_score > best_val:
                        best_val = val_score
                        bad_epochs = 0
                        self.save_embeddings(suffix="best")
                    else:
                        bad_epochs += 1
                        if patience and bad_epochs >= patience:
                            print(f"[EarlyStop] No improvement in {patience} epochs. Stopping at epoch {epoch}.")
                            break

                except Exception as e:
                    print(f"[WARN] evaluation failed: {e}")

            if log_to_wandb:
                try:
                    wandb.log(log_data)
                except Exception as e:
                    print(f"[WARN] wandb.log failed: {e}")

            if epoch % save_every == 0:
                try:
                    self.save_embeddings(suffix=f"epoch{epoch}")
                except Exception as e:
                    print(f"[WARN] save_embeddings failed: {e}")

        if started_wandb and getattr(wandb, "run", None) is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARN] wandb.finish failed: {e}")

    def evaluate(self, triples: list, k: int = 10):
        self.model.eval()
        with torch.no_grad():
            ranks = []
            for s, r, o, conf, src in triples:
                s_t = torch.tensor([s], device=self.device)
                r_t = torch.tensor([r], device=self.device)

                # compute scores for all possible objects
                all_objs = torch.arange(self.model.ent.num_embeddings, device=self.device)
                scores = self.model.score(s_t.repeat(len(all_objs)), r_t.repeat(len(all_objs)), all_objs)
                # lower = better (since it's distance)
                rank = torch.argsort(scores).tolist().index(o) + 1
                ranks.append(rank)

            mrr = np.mean([1.0/r for r in ranks])
            hits1 = np.mean([1 if r <= 1 else 0 for r in ranks])
            hits10 = np.mean([1 if r <= 10 else 0 for r in ranks])
        self.model.train()
        return mrr, hits1, hits10

    def save_embeddings(self, node_out: str = None, rel_out: str = None, suffix: str = ""):
        node_out = Path(node_out) if node_out else (self.kg_dir / f"node_embeddings{('_'+suffix) if suffix else ''}.npy")
        rel_out = Path(rel_out) if rel_out else (self.kg_dir / f"rel_embeddings{('_'+suffix) if suffix else ''}.npy")
        np.save(node_out, self.model.ent.weight.detach().cpu().numpy())
        np.save(rel_out, self.model.rel.weight.detach().cpu().numpy())
        print(f"[KGTransETrainer] saved node embeddings -> {node_out}, rel embeddings -> {rel_out}")
