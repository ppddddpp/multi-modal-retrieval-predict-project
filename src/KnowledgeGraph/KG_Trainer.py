from pathlib import Path
import json
import csv
import os
import numpy as np
import random
from typing import Optional, Tuple, Literal
import datetime
import time
import torch
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim
import torch.nn.functional as F
from .compgcn_conv import CompGCNConv
import wandb
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KG_DIR = BASE_DIR / "knowledge_graph"
KG_DIR.mkdir(parents=True, exist_ok=True)

class BaseKGEModel(nn.Module):
    def __init__(self, n_nodes: int, n_rels: int, emb_dim: int, higher_better: bool = False):
        super().__init__()
        self.ent = nn.Embedding(n_nodes, emb_dim)
        self.rel = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight.data)
        nn.init.xavier_uniform_(self.rel.weight.data)
        self.higher_better = higher_better

    def score(self, s_idx, r_idx, o_idx):
        raise NotImplementedError

class TransEModel(BaseKGEModel):
    def __init__(self, n_nodes, n_rels, emb_dim=200, p_norm=1):
        super().__init__(n_nodes, n_rels, emb_dim)
        self.p = p_norm

    def score(self, s_idx, r_idx, o_idx):
        e_s = self.ent(s_idx)
        e_o = self.ent(o_idx)
        r = self.rel(r_idx)
        diff = e_s + r - e_o
        return torch.norm(diff, p=self.p, dim=1)

class TransHModel(BaseKGEModel):
    def __init__(self, n_nodes, n_rels, emb_dim=200, p_norm=1):
        super().__init__(n_nodes, n_rels, emb_dim)
        self.norm = nn.Embedding(n_rels, emb_dim)  # relation-specific normal vector
        nn.init.xavier_uniform_(self.norm.weight.data)
        self.p = p_norm

    def project(self, e, r_idx):
        n = self.norm(r_idx)
        n = n / torch.norm(n, p=2, dim=1, keepdim=True)
        return e - torch.sum(e * n, dim=1, keepdim=True) * n

    def score(self, s_idx, r_idx, o_idx):
        e_s = self.project(self.ent(s_idx), r_idx)
        e_o = self.project(self.ent(o_idx), r_idx)
        r = self.rel(r_idx)
        diff = e_s + r - e_o
        return torch.norm(diff, p=self.p, dim=1)
    
class RotatEModel(BaseKGEModel):
    def __init__(self, n_nodes, n_rels, emb_dim=200):
        super().__init__(n_nodes, n_rels, emb_dim*2)  # use 2x dim for complex space
        self.emb_dim = emb_dim
        self.higher_better = False

    def score(self, s_idx, r_idx, o_idx):
        # reshape to complex numbers
        e_s = torch.view_as_complex(self.ent(s_idx).view(-1, self.emb_dim, 2))
        e_o = torch.view_as_complex(self.ent(o_idx).view(-1, self.emb_dim, 2))
        r = torch.view_as_complex(self.rel(r_idx).view(-1, self.emb_dim, 2))

        # enforce unit modulus
        r = r / torch.abs(r).clamp(min=1e-9)

        rotated = e_s * r
        diff = rotated - e_o
        return torch.norm(torch.view_as_real(diff), dim=(1, 2))

class CompGCNModel(nn.Module):
    def __init__(self, n_nodes, n_rels, emb_dim=200, num_layers=2,
                 dropout=0.3, bias=True, opn="corr"):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_rels = n_rels
        self.emb_dim = emb_dim
        self.higher_better = False
        self.p = 1 

        # entity & relation embeddings
        self.ent = nn.Embedding(n_nodes, emb_dim)
        self.rel = nn.Embedding(n_rels, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight.data)
        nn.init.xavier_uniform_(self.rel.weight.data)

        # lightweight params object (no recursion)
        params = type("Params", (), {"dropout": dropout, "bias": bias, "opn": opn})()

        # CompGCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                CompGCNConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim,
                    num_rels=n_rels,
                    act=torch.relu,
                    params=params
                )
            )

        # scoring function (TransE-style for simplicity)
        self.higher_better = False

    def forward(self, edge_index, edge_type):
        x = self.ent.weight
        r = self.rel.weight
        for conv in self.layers:
            x, r = conv(x, edge_index, edge_type, r)  # CompGCNConv returns (x, r)
        return x, r

    def score(self, s_idx, r_idx, o_idx, edge_index=None, edge_type=None):
        # forward pass to get updated embeddings
        x, r = self.forward(edge_index, edge_type)
        e_s = x[s_idx]
        e_o = x[o_idx]
        r_vec = r[r_idx]
        diff = e_s + r_vec - e_o
        return torch.norm(diff, p=1, dim=1)

class KGTrainer:
    """
    Class wrapper for training TransE on the triples.csv created by KGBuilder.

    Example:
        tr = KGTrainer(kg_dir="kg", emb_dim=128)
        tr.load_triples()
        tr.train(epochs=10)
        tr.save_embeddings("kg/node_emb.npy", "kg/rel_emb.npy")
    """

    def __init__(self, kg_dir: str = "kg", emb_dim: int = 200, joint_dim: Optional[int] = None, 
                    margin: float = 1.0, lr: float = 1e-3, curated_factor: float = 3.0, 
                    device: Optional[str] = None, model_name: str = "TransE", model_kwargs=None):
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
        self.model_name = model_name

        # these will be filled by load_triples()
        self.node2id = {}
        self.rel2id = {}
        self.triples = []  # tuples (s,r,o,conf,src)
        self.train_triples = []
        self.val_triples = []
        self.model = None
        self.optimizer = None
        self.model_kwargs = model_kwargs or {}

        self._warn_stats = {"count": 0, "time": 0.0}
        
        if self.joint_dim != self.emb_dim:
            self.proj_to_kg = nn.Linear(self.joint_dim, self.emb_dim, bias=False).to(self.device)
        else:
            self.proj_to_kg = nn.Identity()

    def load_maps(self):
        with (self.kg_dir / "node2id.json").open(encoding='utf8') as f:
            self.node2id = json.load(f)
        with (self.kg_dir / "relation2id.json").open(encoding='utf8') as f:
            self.rel2id = json.load(f)

    def load_triples(self, triples_csv: str = None, features_path: str = None):
        # default to kg/triples.csv
        tpath = Path(triples_csv) if triples_csv else self.kg_dir / "triples.csv"
        if not tpath.exists():
            raise FileNotFoundError(tpath)

        # ensure maps loaded
        self.load_maps()
        self.triples = []

        # --- pass 1: count frequencies ---
        counts = {}
        with tpath.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s, r, o = int(row['s_id']), int(row['r_id']), int(row['o_id'])
                key = (s, r, o)
                counts[key] = counts.get(key, 0) + 1

        # --- pass 2: reload with scaling ---
        with tpath.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = int(row['s_id']); r = int(row['r_id']); o = int(row['o_id'])
                except Exception as e:
                    # skip malformed row and log
                    print(f"[KGTrainer] Skipping malformed triple row: {row} ({e})")
                    continue

                conf = float(row.get('confidence', 1.0))
                src = row.get('source', 'extracted')

                # reliability scaling by source
                if src == "curated":
                    conf *= getattr(self, "curated_factor", 3.0)
                elif src == "extracted":  # dataset labels, noisier
                    conf *= 0.7
                elif src in ["mapping", "ontology", "doid", "radlex"]:
                    conf *= 1.0

                # frequency-based noise scaling — use same typed key (ints)
                freq = counts.get((s, r, o), 1)
                conf *= 1.0 / np.log1p(freq)  # downweight overly common triples

                self.triples.append((s, r, o, conf, src))

        print(f"[KGTrainer] loaded {len(self.triples)} triples")

        # train/val split
        random.shuffle(self.triples)
        split = int(0.9 * len(self.triples))
        self.train_triples = self.triples[:split]
        self.val_triples = self.triples[split:]

        # build train-only arrays
        if self.model_name == "CompGCN":
            edge_index = torch.tensor(
                [[t[0] for t in self.train_triples],
                [t[2] for t in self.train_triples]],
                dtype=torch.long, device=self.device
            )
            edge_type = torch.tensor(
                [t[1] for t in self.train_triples],
                dtype=torch.long, device=self.device
            )
            self.edge_index = edge_index
            self.edge_type = edge_type

        self.pos_s = np.array([t[0] for t in self.train_triples], dtype=np.int64)
        self.pos_r = np.array([t[1] for t in self.train_triples], dtype=np.int64)
        self.pos_o = np.array([t[2] for t in self.train_triples], dtype=np.int64)
        self.pos_conf = np.array([t[3] for t in self.train_triples], dtype=np.float32)

        # init model
        n_nodes = len(self.node2id)
        n_rels = len(self.rel2id)

        if self.model_name == "TransE":
            self.model = TransEModel(n_nodes, n_rels, self.emb_dim, **self.model_kwargs).to(self.device)
        elif self.model_name == "TransH":
            self.model = TransHModel(n_nodes, n_rels, self.emb_dim, **self.model_kwargs).to(self.device)
        elif self.model_name == "RotatE":
            self.model = RotatEModel(n_nodes, n_rels, self.emb_dim).to(self.device)
        elif self.model_name == "CompGCN":
            self.model = CompGCNModel(n_nodes, n_rels, self.emb_dim, **self.model_kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown KG model: {self.model_name}")

        if features_path:
            feats_path = Path(features_path)
        else:
            default_pt = (self.kg_dir / "kg_image_feats.pt")
            default_npy = (self.kg_dir / "kg_image_feats.npy")
            if default_pt.exists():
                feats_path = default_pt
            elif default_npy.exists():
                feats_path = default_npy
            else:
                feats_path = None

        if feats_path:
            # replace=True will overwrite embedding vectors. Set replace=False to add on top instead.
            self._inject_image_node_features(feats_path=str(feats_path), replace=True)

        # Create optimizer and include any extra params if present
        kg_lr = getattr(self, "kg_lr", self.lr)
        opt_groups = [{"params": self.model.parameters(), "lr": kg_lr}]

        if getattr(self, "image_feat_proj", None) is not None:
            opt_groups.append({"params": self.image_feat_proj.parameters(), "lr": self.lr * 0.1})

        self.optimizer = optim.Adam(opt_groups)

        s, r, o, *_ = self.train_triples[0]
        s_t = torch.tensor([s], device=self.device)
        r_t = torch.tensor([r], device=self.device)
        o_t = torch.tensor([o], device=self.device)
        ents = torch.arange(self.model.ent.num_embeddings, device=self.device)

        scores_tail = self.batched_scores(s_t, r_t, ents, mode="tail")
        scores_head = self.batched_scores(o_t, r_t, ents, mode="head")

        assert abs(scores_tail[0, o].item() - scores_head[0, s].item()) < 1e-5
        print("Head/tail parity check passed successfully.")

    def _inject_image_node_features(self,
                                    feats_path: str = None,
                                    replace: bool = True,
                                    allow_npy: bool = True,
                                    device: Optional[str] = None):
        """
        Load image feature dict and inject them into self.model.ent.weight for nodes named "image:{id}".
        - replace: True => overwrite embeddings; False => add projected vector on top.
        - Creates self.image_feat_proj if feat_dim != emb_dim and registers it.
        - Returns number of injected nodes.
        """
        if feats_path is None:
            print("[KGTrainer] No feats_path provided, skipping image feature injection.")
            return 0

        dev = self.device if device is None else torch.device(device)
        feats_path = Path(feats_path)
        if not feats_path.exists():
            print(f"[KGTrainer] Image feats file not found: {feats_path}, skipping injection.")
            return 0

        # load dict (support .pt and .npy)
        try:
            if feats_path.suffix == ".pt":
                try:
                    # safe load with weights_only=True (PyTorch >=2.6 default)
                    feats = torch.load(str(feats_path), map_location="cpu", weights_only=True)
                except TypeError:
                    # older torch versions don't know weights_only
                    feats = torch.load(str(feats_path), map_location="cpu")
            else:
                feats = np.load(str(feats_path), allow_pickle=True).item()
        except Exception as e:
            if allow_npy and feats_path.suffix != ".npy":
                alt = feats_path.with_suffix(".npy")
                if alt.exists():
                    feats = np.load(str(alt), allow_pickle=True).item()
                else:
                    print(f"[KGTrainer] Failed to load feats file {feats_path}: {e}")
                    return 0
            else:
                print(f"[KGTrainer] Failed to load feats file {feats_path}: {e}")
                return 0
            
        print(f"[KGTrainer] Loaded {len(feats)} image features from {feats_path}")

        if not isinstance(feats, dict) or len(feats) == 0:
            print(f"[KGTrainer] No features found in {feats_path}, skipping.")
            return 0

        # Infer feature dim
        any_key = next(iter(feats))
        feat_example = feats[any_key]
        if isinstance(feat_example, torch.Tensor):
            feat_dim = feat_example.numel()
        else:
            feat_dim = np.asarray(feat_example).ravel().shape[0]

        emb_dim = self.model.ent.weight.shape[1]

        # create projection if necessary and register it so PyTorch can see its params
        if feat_dim != emb_dim:
            # place projection on trainer device (so subsequent calls are consistent)
            self.image_feat_proj = nn.Linear(feat_dim, emb_dim, bias=True).to(self.device)
            # optional: small init
            nn.init.xavier_uniform_(self.image_feat_proj.weight)
            print(f"[KGTrainer] Created image_feat_proj: {feat_dim} -> {emb_dim}")
        else:
            self.image_feat_proj = None

        injected = 0
        missing_keys = 0
        # inject values
        with torch.no_grad():
            for raw_node_key, vec in feats.items():
                # normalize node key to expected format:
                # allow user to provide keys as "123" or "image:123"
                if isinstance(raw_node_key, (int, float)):
                    node_key = f"image:{int(raw_node_key)}"
                else:
                    node_key = str(raw_node_key)
                    if not node_key.startswith("image:"):
                        # try both possibilities
                        alt = f"image:{node_key}"
                        if alt in self.node2id:
                            node_key = alt

                if node_key not in self.node2id:
                    missing_keys += 1
                    continue

                idx = self.node2id[node_key]
                # convert to tensor on device
                v = torch.tensor(np.asarray(vec).ravel(), dtype=torch.float32, device=self.device)

                # project if needed (projection lives on self.device)
                if self.image_feat_proj is not None:
                    v_proj = self.image_feat_proj(v)  # already on self.device
                    v_write = v_proj.detach()
                else:
                    v_write = v

                # write into embedding table
                with torch.no_grad():
                    if replace:
                        self.model.ent.weight[idx].copy_(v_write)
                    else:
                        # Scale by 0.5 to avoid clipping
                        self.model.ent.weight[idx].add_(v_write * 0.5)

                injected += 1

        with torch.no_grad():
            w = self.model.ent.weight
            self.model.ent.weight.copy_(w / w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6))

        if missing_keys:
            print(f"[KGTrainer] Note: {missing_keys} feature keys did not match any node2id entry and were skipped.")

        print(f"[KGTrainer] Injected image features for {injected} image nodes (out of {len(feats)})")
        return injected

    def train(self,
            epochs: int = 5,
            batch_size: int = 1024,
            normalize: bool = True,
            save_every: int = 1,
            wandb_config: Optional[dict] = None,
            log_to_wandb: bool = True,
            patience: Optional[int] = None,
            metric: str = "mrr",
            loss_type: str = "logsigmoid",
            negative_size: int = 32,
            advance_temp: float = 1.0,
            use_amp: Optional[bool] = None,
            seed: Optional[int] = None,
            progress_bar_for_eval: bool = True
            ):
        """
        Train the Knowledge Graph model using the provided triples.

        Args:
            epochs (int, optional): Number of epochs to train for. Defaults to 5.
            batch_size (int, optional): Batch size to use during training. Defaults to 1024.
            normalize (bool, optional): Whether to normalize the embeddings after training. Defaults to True.
            save_every (int, optional): Number of epochs between saving the model's embeddings. Defaults to 1.
            wandb_config (dict, optional): Configuration for Weights and Biases (WandB) logging. Defaults to None.
            log_to_wandb (bool, optional): Whether to log training metrics to WandB. Defaults to True.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to None.
            metric (str, optional): Metric to use for early stopping. Defaults to "mrr".
            loss_type (str, optional): Type of loss function to use. Defaults to "logsigmoid".
            negative_size (int, optional): Number of negative samples to use for each positive sample. Defaults to 32.
            advance_temp (float, optional): Temperature to use for adversarial weighting. Defaults to 1.0.
            use_amp (bool, optional): Whether to use automatic mixed precision training. Defaults to None.
            seed (int, optional): Seed to use for reproducibility. Defaults to None.
            progress_bar_for_eval (bool, optional): Whether to display a progress bar during evaluation. Defaults to True.

        Returns:
            int: Number of epochs trained for.
        """
        if self.model is None:
            raise RuntimeError("Call load_triples() first.")
        self.model.train()

        # reproducibility for fair comparison
        if seed is not None:
            import random as _py_random
            _py_random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if use_amp is None:
            use_amp = torch.cuda.is_available()
        use_amp = bool(use_amp) and torch.cuda.is_available()

        scaler = amp.GradScaler(enabled=use_amp)

        n = len(self.pos_s)
        steps = max(1, (n + batch_size - 1) // batch_size)

        # wandb init bookkeeping
        started_wandb = False
        if log_to_wandb:
            try:
                if getattr(wandb, "run", None) is None:
                    run_name = wandb_config.get("name") if wandb_config and "name" in wandb_config \
                            else f"kg_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        best_kg_metrics = {"mrr": -float("inf"), "hits1": -float("inf"), "hits5": -float("inf"), "hits10": -float("inf")}
        best_epoch = -1

        neg_size = negative_size if negative_size > 0 else 32
        adv_temp = advance_temp if advance_temp > 0.0 else 1.0

        for epoch in tqdm(range(1, epochs + 1), desc="KG Train", unit="epoch"):
            self.model.train()
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            epoch_loss = 0.0

            self._cached_x_all = None
            self._cached_r_all = None
            x_all = None
            r_all = None

            # instrumentation
            print_every = 10  # how often to print status (batches)
            batch_times = []
            moving_avg = None

            for step in range(steps):
                start = step * batch_size
                end = min((step + 1) * batch_size, n)
                if start >= end:
                    continue
                batch_idx = idxs[start:end]

                batch_t0 = time.time()

                s_batch = torch.from_numpy(self.pos_s[batch_idx]).long().to(self.device, non_blocking=True)
                r_batch = torch.from_numpy(self.pos_r[batch_idx]).long().to(self.device, non_blocking=True)
                o_batch = torch.from_numpy(self.pos_o[batch_idx]).long().to(self.device, non_blocking=True)
                conf_batch = torch.from_numpy(self.pos_conf[batch_idx]).float().to(self.device, non_blocking=True)

                # --- positive scores ---
                with amp.autocast("cuda", enabled=use_amp):
                    # compute positive scores (handles CompGCN precomputed path)
                    if self.model_name == "CompGCN":
                        if x_all is None:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_fwd = time.time()
                            x_tmp, r_tmp = self.model(self.edge_index, self.edge_type)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_fwd = time.time() - t_fwd
                            self._warn_stats["count"] += 1
                            self._warn_stats["time"] += t_fwd
                            e_s = x_tmp[s_batch]
                            e_o = x_tmp[o_batch]
                            r_vec = r_tmp[r_batch]
                        else:
                            e_s = x_all[s_batch]
                            e_o = x_all[o_batch]
                            r_vec = r_all[r_batch]
                        pos_scores = torch.norm(e_s + r_vec - e_o, p=1, dim=1)
                    else:
                        pos_scores = self.model.score(s_batch, r_batch, o_batch)

                    # negative sampling 
                    B = len(batch_idx)
                    rand_nodes = torch.randint(0, self.model.ent.num_embeddings, (B, neg_size), device=self.device, dtype=torch.long)

                    neg_s = s_batch.unsqueeze(1).expand(-1, neg_size).clone()  # (B, neg_size)
                    neg_o = o_batch.unsqueeze(1).expand(-1, neg_size).clone()

                    corrupt_mask = (torch.rand(B, neg_size, device=self.device) < 0.5)
                    neg_s[corrupt_mask] = rand_nodes[corrupt_mask]
                    neg_o[~corrupt_mask] = rand_nodes[~corrupt_mask]

                    # flattened views for scoring
                    s_flat = neg_s.view(-1)
                    r_flat = r_batch.unsqueeze(1).expand(-1, neg_size).contiguous().view(-1)
                    o_flat = neg_o.view(-1)

                    # negative scoring (vectorized, uses precomputed x_all when available)
                    if self.model_name == "CompGCN":
                        if x_all is None:
                            neg_scores_all = self.model.score(s_flat, r_flat, o_flat,
                                                            self.edge_index, self.edge_type).view(B, neg_size)
                        else:
                            e_s_neg = x_all[s_flat]
                            e_o_neg = x_all[o_flat]
                            r_neg = r_all[r_flat]
                            neg_scores_all = torch.norm(e_s_neg + r_neg - e_o_neg, p=1, dim=1).view(B, neg_size)
                    else:
                        # plain models: score is vectorized, accept flattened index tensors
                        neg_scores_all = self.model.score(s_flat, r_flat, o_flat).view(B, neg_size)

                    # adversarial weighting + loss
                    if getattr(self.model, "higher_better", False):
                        weights = F.softmax(neg_scores_all / adv_temp, dim=1).detach()
                    else:
                        weights = F.softmax(-neg_scores_all / adv_temp, dim=1).detach()

                    neg_score = (weights * neg_scores_all).sum(dim=1)
                    
                    if loss_type == "logsigmoid":
                        loss = - (F.logsigmoid(neg_score - pos_scores) * conf_batch).mean()
                    elif loss_type == "margin":
                        margin = getattr(self, "margin", 1.0)
                        loss = F.relu(margin + pos_scores - neg_score).mean()
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")

                # ---------- Backward with GradScaler ----------
                self.optimizer.zero_grad()

                if not getattr(loss, "requires_grad", False):
                    # helpful debug prints
                    print(f"[ERROR] loss.requires_grad=False; detaching info:")
                    try:
                        print("  pos_scores.requires_grad =", getattr(pos_scores, "requires_grad", None))
                        print("  neg_scores_all.requires_grad =", getattr(neg_scores_all, "requires_grad", None))
                    except Exception:
                        pass
                    # clear cache to avoid future silent reuse
                    self._cached_x_all = None
                    self._cached_r_all = None
                    raise RuntimeError(
                        "Loss has requires_grad=False. Likely cause: using cached embeddings "
                        "computed with torch.no_grad() during training. See KGTrainer fixes."
                    )

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # bookkeeping / logging (unchanged)
                epoch_loss += float(loss.item())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_t1 = time.time()
                batch_time = batch_t1 - batch_t0
                if moving_avg is None:
                    moving_avg = batch_time
                else:
                    moving_avg = 0.9 * moving_avg + 0.1 * batch_time

                # --- debugging scalars ---
                pos_mean = float(pos_scores.mean().detach().cpu().item())
                neg_mean = float(neg_scores_all.mean().detach().cpu().item())
                neg_weight_entropy = float((- (weights * (weights + 1e-12).log()).sum(dim=1)).mean().detach().cpu().item())

                if log_to_wandb and ((step + 1) % print_every == 0 or (step + 1) == steps):
                    try:
                        wandb.log({
                            "kg/pos_score_mean": pos_mean,
                            "kg/neg_score_mean": neg_mean,
                            "kg/neg_weight_entropy": neg_weight_entropy,
                            "kg/epoch": epoch,
                            "kg/batch_idx": step+1,
                        })
                    except Exception as e:
                        print(f"[WARN] wandb.log failed (pos/neg): {e}")

                if (step + 1) % print_every == 0 or (step + 1) == steps:
                    batches_left = steps - (step + 1)
                    eta = batches_left * moving_avg
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    print(f"[KGTrainer] epoch {epoch} batch {step+1}/{steps} — loss={loss.item():.4f} "
                        f"avg_batch_time={moving_avg:.2f}s ETA={eta/60:.2f}min GPU_mem={gpu_mem:.2f}GB")
                    if log_to_wandb:
                        try:
                            wandb.log({
                                "kg/batch_loss": loss.item(),
                                "kg/batch_time": batch_time,
                                "kg/avg_batch_time": moving_avg,
                                "kg/epoch": epoch,
                                "kg/batch_idx": step+1,
                                "kg/use_amp": bool(use_amp)
                            })
                        except Exception as e:
                            print(f"[WARN] wandb.log failed: {e}")

            if self._warn_stats["count"] > 0:
                avg = self._warn_stats["time"] / self._warn_stats["count"]
                print(f"[KGTrainer] Slow forward fallback happened "
                    f"{self._warn_stats['count']} times "
                    f"(avg {avg:.3f}s)")
                self._warn_stats = {"count": 0, "time": 0.0}

            # normalize embeddings
            if normalize:
                with torch.no_grad():
                    w = self.model.ent.weight
                    denom = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
                    self.model.ent.weight.copy_(w / denom)

            with torch.no_grad():
                r = self.model.rel.weight
                denom_r = r.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
                self.model.rel.weight.copy_(r / denom_r)

            avg_loss = epoch_loss / max(1, steps)
            print(f"[KGTrainer] Epoch {epoch}/{epochs} avg_loss={avg_loss:.6f}")

            # prepare logging payload
            log_data = {"kg/epoch": epoch, "kg/loss": avg_loss}

            # evaluate
            if hasattr(self, "val_triples") and len(self.val_triples) > 0:
                try:
                    mrr, hits1, hits5, hits10 = self.evaluate(self.val_triples, k=10, use_amp=use_amp, progress=progress_bar_for_eval)

                    # --- Track best metrics ---
                    if mrr > best_kg_metrics["mrr"]:
                        best_kg_metrics["mrr"] = mrr
                        best_epoch = epoch + 1
                        if log_to_wandb:
                            wandb.log({"kg/best_mrr": mrr, "kg/best_mrr_epoch": best_epoch})

                    if hits1 > best_kg_metrics["hits1"]:
                        best_kg_metrics["hits1"] = hits1
                        if log_to_wandb:
                            wandb.log({"kg/best_hits1": hits1, "kg/best_hits1_epoch": epoch + 1})

                    if hits5 > best_kg_metrics["hits5"]:
                        best_kg_metrics["hits5"] = hits5
                        if log_to_wandb:
                            wandb.log({"kg/best_hits5": hits5, "kg/best_hits5_epoch": epoch + 1})

                    if hits10 > best_kg_metrics["hits10"]:
                        best_kg_metrics["hits10"] = hits10
                        if log_to_wandb:
                            wandb.log({"kg/best_hits10": hits10, "kg/best_hits10_epoch": epoch + 1})

                    print(f"[Eval] MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@10={hits10:.4f}")
                    log_data.update({
                        "kg/val_mrr": mrr,
                        "kg/val_hits1": hits1,
                        "kg/val_hits5": hits5,
                        "kg/val_hits10": hits10,
                    })

                    # early stopping
                    val_score = {"mrr": mrr, "hits1": hits1, "hits5": hits5, "hits10": hits10}[metric]
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
                    try:
                        self.model.train()
                    except Exception:
                        pass
                    self._cached_x_all = None
                    self._cached_r_all = None

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

        if log_to_wandb:
            for k, v in best_kg_metrics.items():
                wandb.run.summary[f"kg_best_{k}"] = v
            wandb.run.summary["kg_best_epoch"] = best_epoch

        # --- Save best metrics as JSON ---
        best_path = BASE_DIR / "best"
        if not best_path.exists():
            best_path.mkdir(parents=True)
        best_json_path = best_path / "best_metrics.json"
        best_payload = {
            "best_epoch": best_epoch,
            "metric_used": metric,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        best_payload.update(best_kg_metrics)

        try:
            with best_json_path.open("w", encoding="utf8") as f:
                json.dump(best_payload, f, indent=2)
            print(f"[KGTrainer] Saved best metrics -> {best_json_path}")
        except Exception as e:
            print(f"[WARN] Failed to save best metrics JSON: {e}")

    def probe_max_eval_batch(self,
                            s_batch: torch.Tensor,
                            r_batch: torch.Tensor,
                            candidates: torch.LongTensor,
                            use_amp: Optional[bool] = None,
                            start: int = 1024,
                            max_batch: int = 1 << 20,
                            safety_factor: float = 0.9,
                            max_trials: int = 20,
                            target_util: float = 0.2) -> int:
        """
        Find the largest candidate-chunk size that can be scored without OOM,
        for a batch of (s, r) queries at once, and then scale it up so that
        GPU memory utilization is at least `target_util`.

        Args:
            s_batch, r_batch: tensors of shape (B,) describing multiple queries.
            candidates: 1D LongTensor (all entity IDs).
            use_amp: whether to use autocast (None -> auto if CUDA available).
            start: initial candidate batch size to try.
            max_batch: absolute upper limit.
            safety_factor: fraction to apply to the found maximum.
            max_trials: number of probing attempts.
            target_util: target GPU memory utilization (e.g. 0.2 => 20%).

        Returns:
            suggested candidate batch size (int).
        """
        device = self.device
        if use_amp is None:
            use_amp = torch.cuda.is_available()
        use_amp = bool(use_amp) and torch.cuda.is_available()

        if not isinstance(candidates, torch.Tensor):
            candidates = torch.as_tensor(candidates, dtype=torch.long, device=device)
        else:
            candidates = candidates.to(device)

        # CPU fallback
        if not torch.cuda.is_available():
            return min(4096, candidates.size(0))

        B = s_batch.size(0)

        def try_size(sz: int) -> bool:
            sz = min(sz, candidates.size(0))
            if sz <= 0:
                return False
            cand_chunk = candidates[:sz]

            try:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    with amp.autocast("cuda", enabled=use_amp):
                        if self.model_name == "CompGCN":
                            x_all = getattr(self, "_cached_x_all", None)
                            r_all = getattr(self, "_cached_r_all", None)

                            if x_all is not None and r_all is not None:
                                e_s = x_all[s_batch]        # (B, d)
                                r_vec = r_all[r_batch]      # (B, d)
                                e_o = x_all[cand_chunk]     # (C, d)
                                diff = e_s.unsqueeze(1) + r_vec.unsqueeze(1) - e_o.unsqueeze(0)
                                _ = torch.norm(diff, p=1, dim=2)
                            else:
                                e_s = self.model.ent(s_batch)   # (B, d)
                                r_vec = self.model.rel(r_batch) # (B, d)
                                e_o = self.model.ent(cand_chunk) # (C, d)
                                diff = e_s.unsqueeze(1) + r_vec.unsqueeze(1) - e_o.unsqueeze(0)
                                _ = torch.norm(diff, p=1, dim=2)
                        else:
                            e_s = self.model.ent(s_batch)       # (B, d)
                            r_vec = self.model.rel(r_batch)     # (B, d)
                            e_o = self.model.ent(cand_chunk)    # (C, d)
                            diff = e_s.unsqueeze(1) + r_vec.unsqueeze(1) - e_o.unsqueeze(0)
                            _ = torch.norm(diff, p=1, dim=2)
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return False
                raise

        # exponential growth until fail
        lo, hi = 0, None
        cur = start
        trials = 0
        while trials < max_trials:
            trials += 1
            ok = try_size(cur)
            if ok:
                lo = cur
                if cur >= max_batch or cur >= candidates.size(0):
                    hi = cur
                    break
                nxt = min(cur * 2, max_batch)
                if nxt == cur:
                    hi = cur
                    break
                cur = nxt
            else:
                hi = cur
                break

        # fallback if nothing worked
        if lo == 0 and (hi is None or hi > 1):
            for fallback in [512, 256, 64]:
                if try_size(fallback):
                    lo = fallback
                    break

        if hi is None:
            hi = min(max_batch, candidates.size(0))

        # refine with binary search
        while lo + 1 < hi and trials < max_trials:
            trials += 1
            mid = (lo + hi) // 2
            if try_size(mid):
                lo = mid
            else:
                hi = mid

        max_ok = max(1, lo)
        suggested = max(1, int(max_ok * safety_factor))
        suggested = min(suggested, candidates.size(0))

        # --- new part: bump to reach ≥ target_util ---
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        used_percent = used / total if total > 0 else 0.0

        if used_percent < target_util:
            factor = target_util / max(used_percent, 1e-6)
            bumped = int(suggested * factor)
            bumped = min(bumped, candidates.size(0), max_batch)
            if bumped > suggested:
                print(f"[KGTrainer] Bumping cand_batch_size {suggested} -> {bumped} "
                      f"to reach more than {target_util*100:.0f}% GPU util "
                      f"(current {used_percent*100:.1f}%)")
                suggested = bumped

        return suggested

    def batched_scores(self,
                    s_t: torch.Tensor,
                    r_t: torch.Tensor,
                    candidates: torch.LongTensor,
                    batch_size: int = 2048,
                    use_amp: Optional[bool] = None,
                    x_all: Optional[torch.Tensor] = None,
                    r_all: Optional[torch.Tensor] = None,
                    show_progress: bool = False,
                    mode: Literal["tail", "head"] = "tail"):
        """
        Compute scores for queries in batch mode.

        Args:
            s_t: for mode="tail": subject indices (B,)
                for mode="head": object indices (B,)  <-- IMPORTANT: interpret s_t as objects when mode="head"
            r_t: relation indices (B,)
            candidates: 1D tensor of candidate entity ids (N,)
            batch_size: candidate chunk size
            x_all, r_all: optional precomputed propagated embeddings (CompGCN fast-path)
            mode: "tail" -> score (s, r, candidates)  (default)
                "head" -> score (candidates, r, o) where s_t is treated as o (objects)
        Returns:
            Tensor of shape (B, N) with distances/scores.
        """

        device = self.device
        if use_amp is None:
            use_amp = torch.cuda.is_available()
        use_amp = bool(use_amp) and torch.cuda.is_available()

        # ensure tensors on device
        if not isinstance(candidates, torch.Tensor):
            candidates = torch.as_tensor(candidates, dtype=torch.long, device=device)
        else:
            candidates = candidates.to(device).long()

        if not isinstance(s_t, torch.Tensor):
            s_t = torch.as_tensor(s_t, dtype=torch.long, device=device)
        else:
            s_t = s_t.to(device).long()

        if not isinstance(r_t, torch.Tensor):
            r_t = torch.as_tensor(r_t, dtype=torch.long, device=device)
        else:
            r_t = r_t.to(device).long()

        B = s_t.numel()
        N = candidates.numel()
        if N == 0:
            return torch.empty((B, 0), device=device)

        iter_ranges = range(0, N, batch_size)
        if show_progress:
            iter_ranges = tqdm(iter_ranges, desc="batched_scores", unit="chunk")

        results = []
        self.model.eval()
        with torch.no_grad():
            with amp.autocast("cuda", enabled=use_amp):
                for start in iter_ranges:
                    end = min(start + batch_size, N)
                    cand_chunk = candidates[start:end]  # (C,)
                    C = cand_chunk.numel()

                    # --- Fast path: have precomputed propagated embeddings (CompGCN) ---
                    if self.model_name == "CompGCN" and x_all is not None and r_all is not None:
                        if mode == "tail":
                            # tail: scores for (s_t, r_t, cand_chunk)
                            e_s = x_all[s_t]           # (B, d)
                            r_vec = r_all[r_t]         # (B, d)
                            e_o = x_all[cand_chunk]    # (C, d)
                            diff = e_s.unsqueeze(1) + r_vec.unsqueeze(1) - e_o.unsqueeze(0)  # (B, C, d)
                        else:  # mode == "head"
                            # head: scores for (cand_chunk, r_t, o_batch) where s_t is o_batch
                            e_o_batch = x_all[s_t]     # (B, d)  (s_t holds o_batch in head-mode)
                            r_vec = r_all[r_t]         # (B, d)
                            e_c = x_all[cand_chunk]    # (C, d)
                            diff = e_c.unsqueeze(0) + r_vec.unsqueeze(1) - e_o_batch.unsqueeze(1)  # (B, C, d)

                        p = getattr(self.model, "p", 2)
                        scores = torch.norm(diff, p=p, dim=2)  # (B, C)
                        results.append(scores)
                        continue

                    # --- No-cache CompGCN path: call model.score on flattened lists ---
                    if self.model_name == "CompGCN" and (x_all is None or r_all is None):
                        if mode == "tail":
                            # want scores for (s_i, r_i, o_c) for each i in [0..B-1], c in [0..C-1]
                            s_rep = s_t.repeat_interleave(C)            # (B*C,)
                            r_rep = r_t.repeat_interleave(C)            # (B*C,)
                            o_rep = cand_chunk.repeat(B)                # (B*C,)
                        else:
                            # mode == "head": s_t is o_batch; want scores (cand_c, r_i, o_i)
                            s_rep = cand_chunk.repeat(B)                # (B*C,)
                            r_rep = r_t.repeat_interleave(C)           # (B*C,)
                            o_rep = s_t.repeat_interleave(C)           # (B*C,)

                        scores_flat = self.model.score(s_rep, r_rep, o_rep,
                                                    getattr(self, "edge_index", None),
                                                    getattr(self, "edge_type", None))
                        # reshape to (B, C) in query-major order
                        results.append(scores_flat.view(B, -1))
                        continue

                    # --- Vanilla models (no GCN) ---
                    if mode == "tail":
                        e_s = self.model.ent(s_t)        # (B, d)
                        r_vec = self.model.rel(r_t)      # (B, d)
                        e_o = self.model.ent(cand_chunk) # (C, d)
                        diff = e_s.unsqueeze(1) + r_vec.unsqueeze(1) - e_o.unsqueeze(0)  # (B, C, d)
                    else:
                        # head mode: s_t is o_batch
                        e_o_batch = self.model.ent(s_t)      # (B, d)
                        r_vec = self.model.rel(r_t)          # (B, d)
                        e_c = self.model.ent(cand_chunk)     # (C, d)
                        diff = e_c.unsqueeze(0) + r_vec.unsqueeze(1) - e_o_batch.unsqueeze(1)  # (B, C, d)

                    p = getattr(self.model, "p", 2)
                    scores = torch.norm(diff, p=p, dim=2)  # (B, C)
                    results.append(scores)

        out = torch.cat(results, dim=1)  # (B, N)
        if out.shape != (B, N):
            raise RuntimeError(f"batched_scores: got {out.shape}, expected ({B},{N})")

        self.model.train()
        return out

    def evaluate(self, triples: list, k: int = 10, triple_batch_size: int = 64,
                cand_batch_size: Optional[int] = None, use_amp: Optional[bool] = None,
                progress: bool = True):
        """
        Evaluate model on given triples.

        Args:
            triples: List of triples to evaluate (s, r, o, _, _)
            k: int = 10, number of top scores to consider for ranking
            triple_batch_size: int = 64, batch size for triples
            cand_batch_size: Optional[int] = None, batch size for candidate entities
            use_amp: Optional[bool] = None, whether to use AMP (auto mixed precision)
            progress: bool = True, whether to show progress bar

        Returns:
            mrr: float, mean reciprocal rank
            hits1: float, proportion of correct entities in top 1
            hits5: float, proportion of correct entities in top 5
            hits10: float, proportion of correct entities in top 10

        Notes:
            - If cand_batch_size is not set, it is automatically probed in the range [4096, 1 << 16]
            - If use_amp is not set, it is automatically set to True if CUDA is available
        """
        all_known = set((s, r, o) for s, r, o, _, _ in self.triples)
        self.model.eval()
        ranks = []
        device = self.device

        print(f"[KGTrainer] Starting evaluation on {len(triples)} triples; "
            f"n_nodes={self.model.ent.num_embeddings}; "
            f"triple_batch_size={triple_batch_size}, cand_batch_size={cand_batch_size}")
        t_eval_start = time.time()

        with torch.no_grad():
            x_all = getattr(self, "_cached_x_all", None)
            r_all = getattr(self, "_cached_r_all", None)

            if self.model_name == "CompGCN" and (x_all is None or r_all is None):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time()
                    x_all, r_all = self.model(self.edge_index, self.edge_type)  # on device
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_fwd = time.time() - t0
                # cache them for reuse during this evaluate call (and optionally future calls)
                self._cached_x_all = x_all
                self._cached_r_all = r_all
                # optional debug print
                print(f"[KGTrainer] Computed CompGCN propagated embeddings for evaluate() in {t_fwd:.3f}s")

            all_entities = torch.arange(self.model.ent.num_embeddings, device=device)

            # --- probe candidate batch size automatically if not set ---
            if cand_batch_size is None and len(triples) > 0:
                probe_batch = triples[:min(triple_batch_size, len(triples))]
                s_batch = torch.tensor([s for s, _, _, _, _ in probe_batch], device=device)
                r_batch = torch.tensor([r for _, r, _, _, _ in probe_batch], device=device)

                cand_batch_size = self.probe_max_eval_batch(
                    s_batch, r_batch, all_entities,
                    use_amp=use_amp, start=4096, max_batch=1 << 16
                )
                print(f"[KGTrainer] Probed cand_batch_size={cand_batch_size}")

            # --- iterate in triple batches ---
            iterator = range(0, len(triples), triple_batch_size)
            if progress:
                iterator = tqdm(iterator, desc="KG Eval (multi)", unit="batch")

            for start in iterator:
                batch = triples[start:start + triple_batch_size]
                s_batch = torch.tensor([s for s, _, _, _, _ in batch], device=device)
                r_batch = torch.tensor([r for _, r, _, _, _ in batch], device=device)
                o_batch = torch.tensor([o for _, _, o, _, _ in batch], device=device)

                # --- Tail prediction: (s, r, ?)
                scores_tail = self.batched_scores(s_batch, r_batch, all_entities,
                        batch_size=cand_batch_size, use_amp=use_amp, mode="tail",
                        x_all=x_all, r_all=r_all) # (B, num_entities)

                for i, (s, r, o, _, _) in enumerate(batch):
                    known_tails = [obj for (ss, rr, obj) in all_known if ss == s and rr == r and obj != o]
                    if known_tails:
                        mask = torch.tensor(known_tails, device=device)
                        if getattr(self.model, "higher_better", False):
                            scores_tail[i].index_fill_(0, mask, -float("inf"))
                        else:
                            scores_tail[i].index_fill_(0, mask, float("inf"))

                    sorted_idx = torch.argsort(
                        scores_tail[i],
                        descending=getattr(self.model, "higher_better", False)
                    )
                    rank_tail = (sorted_idx == o).nonzero(as_tuple=False).item() + 1
                    ranks.append(rank_tail)

                # --- Head prediction: (?, r, o)
                scores_head = self.batched_scores(o_batch, r_batch, all_entities,
                    batch_size=cand_batch_size, use_amp=use_amp, mode="head",
                    x_all=x_all, r_all=r_all) # (B, num_entities)

                for i, (s, r, o, _, _) in enumerate(batch):
                    known_heads = [ss for (ss, rr, obj) in all_known if rr == r and obj == o and ss != s]
                    if known_heads:
                        mask = torch.tensor(known_heads, device=device)
                        if getattr(self.model, "higher_better", False):
                            scores_head[i].index_fill_(0, mask, -float("inf"))
                        else:
                            scores_head[i].index_fill_(0, mask, float("inf"))

                    sorted_idx = torch.argsort(
                        scores_head[i],
                        descending=getattr(self.model, "higher_better", False)
                    )
                    rank_head = (sorted_idx == s).nonzero(as_tuple=False).item() + 1
                    ranks.append(rank_head)

        # --- compute metrics ---
        mrr = np.mean([1.0 / r for r in ranks]) if ranks else 0.0
        hits1 = np.mean([r <= 1 for r in ranks]) if ranks else 0.0
        hits5 = np.mean([r <= 5 for r in ranks]) if ranks else 0.0
        hits10 = np.mean([r <= 10 for r in ranks]) if ranks else 0.0

        t_eval_total = time.time() - t_eval_start
        print(f"[KGTrainer] Evaluation finished in {t_eval_total:.1f}s — "
            f"mrr={mrr:.4f}, hits1={hits1:.4f}, hits5={hits5:.4f}, hits10={hits10:.4f}")

        self.model.train()
        return mrr, hits1, hits5, hits10

    def save_embeddings(self, node_out: str = None, rel_out: str = None, suffix: str = ""):
        node_out = Path(node_out) if node_out else (self.kg_dir / f"node_embeddings{('_'+suffix) if suffix else ''}.npy")
        rel_out = Path(rel_out) if rel_out else (self.kg_dir / f"rel_embeddings{('_'+suffix) if suffix else ''}.npy")
        meta_out = self.kg_dir / f"embeddings_meta{('_'+suffix) if suffix else ''}.json"

        ent_w = self.model.ent.weight.detach().cpu().numpy()
        rel_w = self.model.rel.weight.detach().cpu().numpy()

        if self.model_name == "CompGCN":
            # run graph forward to get propagated embeddings
            x, r = self.model(self.edge_index, self.edge_type)
            ent_w = x.detach().cpu().numpy()
            rel_w = r.detach().cpu().numpy()
            np.save(node_out, ent_w)
            np.save(rel_out, rel_w)
            # save metadata as well (important for later restores)
            meta = {
                "model_name": self.model_name,
                "emb_dim": self.emb_dim,
                "n_nodes": ent_w.shape[0],
                "n_rels": rel_w.shape[0],
                "ent_shape": list(ent_w.shape),
                "rel_shape": list(rel_w.shape),
                "higher_better": getattr(self.model, "higher_better", False),
            }
            with meta_out.open("w") as f:
                json.dump(meta, f, indent=2)
            print(f"[KGTrainer] saved CompGCN propagated embeddings -> {node_out}, {rel_out}")
            print(f"[KGTrainer] saved metadata -> {meta_out}")
            return  # skip the rest of save logic

        if self.model_name == "RotatE":
            # reshape to (n, emb_dim, 2) -> complex array
            n_ent = ent_w.reshape(ent_w.shape[0], self.model.emb_dim, 2)
            n_rel = rel_w.reshape(rel_w.shape[0], self.model.emb_dim, 2)

            n_ent = n_ent[..., 0] + 1j * n_ent[..., 1]
            n_rel = n_rel[..., 0] + 1j * n_rel[..., 1]

            np.save(node_out, n_ent)
            np.save(rel_out, n_rel)
            print(f"[KGTrainer] saved RotatE complex embeddings -> {node_out}, {rel_out}")
        else:
            np.save(node_out, ent_w)
            np.save(rel_out, rel_w)
            print(f"[KGTrainer] saved embeddings -> {node_out}, {rel_out}")

        # save metadata with full shapes
        meta = {
            "model_name": self.model_name,
            "emb_dim": self.emb_dim,
            "n_nodes": ent_w.shape[0],
            "n_rels": rel_w.shape[0],
            "ent_shape": list(ent_w.shape),
            "rel_shape": list(rel_w.shape),
            "higher_better": getattr(self.model, "higher_better", False),
        }
        with meta_out.open("w") as f:
            json.dump(meta, f, indent=2)
        print(f"[KGTrainer] saved metadata -> {meta_out}")

    def _resize_embeddings(self, arr: np.ndarray, target_shape: Tuple[int, int], name: str) -> np.ndarray:
        """Resize embeddings array (pad or truncate) to match target shape."""
        out = np.zeros(target_shape, dtype=arr.dtype)
        min_rows = min(arr.shape[0], target_shape[0])
        min_cols = min(arr.shape[1], target_shape[1])
        out[:min_rows, :min_cols] = arr[:min_rows, :min_cols]

        if arr.shape[0] < target_shape[0] or arr.shape[1] < target_shape[1]:
            print(f"[WARN] {name} embeddings padded from {arr.shape} -> {target_shape}")
            # Xavier init for padded region
            fan_in, fan_out = target_shape[1], target_shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            pad = np.random.uniform(-limit, limit, size=target_shape).astype(arr.dtype)
            out[min_rows:, :] = pad[min_rows:, :]
            out[:, min_cols:] = pad[:, min_cols:]
        elif arr.shape != target_shape:
            print(f"[WARN] {name} embeddings truncated from {arr.shape} -> {target_shape}")
        return out

    def load_embeddings(self, node_in: str = None, rel_in: str = None, suffix: str = "", strict_shapes: bool = False):
        node_in = Path(node_in) if node_in else (self.kg_dir / f"node_embeddings{('_'+suffix) if suffix else ''}.npy")
        rel_in = Path(rel_in) if rel_in else (self.kg_dir / f"rel_embeddings{('_'+suffix) if suffix else ''}.npy")
        meta_in = self.kg_dir / f"embeddings_meta{('_'+suffix) if suffix else ''}.json"

        # load metadata if available
        meta = {}
        if meta_in.exists():
            with meta_in.open() as f:
                meta = json.load(f)
            print(f"[KGTrainer] loaded metadata <- {meta_in}")
            self.model.higher_better = meta.get("higher_better", False)
        else:
            print(f"[KGTrainer] WARNING: no metadata found, using defaults")
            self.model.higher_better = False

        if self.model_name == "RotatE":
            ent_w = np.load(node_in)
            rel_w = np.load(rel_in)

            assert np.iscomplexobj(ent_w), "RotatE node embeddings must be complex"
            assert np.iscomplexobj(rel_w), "RotatE rel embeddings must be complex"

            # convert back to stacked real
            ent_real = np.stack([ent_w.real, ent_w.imag], axis=-1).reshape(ent_w.shape[0], -1)
            rel_real = np.stack([rel_w.real, rel_w.imag], axis=-1).reshape(rel_w.shape[0], -1)

            if strict_shapes:
                if list(ent_real.shape) != list(self.model.ent.weight.shape):
                    raise ValueError(f"Entity shape mismatch: {ent_real.shape} vs {tuple(self.model.ent.weight.shape)}")
                if list(rel_real.shape) != list(self.model.rel.weight.shape):
                    raise ValueError(f"Relation shape mismatch: {rel_real.shape} vs {tuple(self.model.rel.weight.shape)}")
            else:
                ent_real = self._resize_embeddings(ent_real, self.model.ent.weight.shape, "RotatE nodes")
                rel_real = self._resize_embeddings(rel_real, self.model.rel.weight.shape, "RotatE rels")

            with torch.no_grad():
                self.model.ent.weight.copy_(torch.tensor(ent_real, dtype=torch.float32, device=self.device))
                self.model.rel.weight.copy_(torch.tensor(rel_real, dtype=torch.float32, device=self.device))
            print(f"[KGTrainer] loaded RotatE complex embeddings <- {node_in}, {rel_in}")

        else:
            ent_w = np.load(node_in)
            rel_w = np.load(rel_in)

            if strict_shapes:
                if list(ent_w.shape) != list(self.model.ent.weight.shape):
                    raise ValueError(f"Entity shape mismatch: {ent_w.shape} vs {tuple(self.model.ent.weight.shape)}")
                if list(rel_w.shape) != list(self.model.rel.weight.shape):
                    raise ValueError(f"Relation shape mismatch: {rel_w.shape} vs {tuple(self.model.rel.weight.shape)}")
            else:
                ent_w = self._resize_embeddings(ent_w, self.model.ent.weight.shape, "nodes")
                rel_w = self._resize_embeddings(rel_w, self.model.rel.weight.shape, "rels")

            with torch.no_grad():
                self.model.ent.weight.copy_(torch.tensor(ent_w, dtype=torch.float32, device=self.device))
                self.model.rel.weight.copy_(torch.tensor(rel_w, dtype=torch.float32, device=self.device))
            print(f"[KGTrainer] loaded embeddings <- {node_in}, {rel_in}")
