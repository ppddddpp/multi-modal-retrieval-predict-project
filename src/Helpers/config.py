from dataclasses import dataclass, field, fields, asdict
import yaml

@dataclass
class Config:
    # Training parameters
    epochs: int = 50
    patience: int = 10
    batch_size: int = 1
    lr: float = 2e-5

    # Model parameters
    num_fusion_layers: int = 3
    use_focal: bool = False
    use_hybrid: bool = True
    image_backbone: str = "swin"
    fusion_type: str = "cross"
    joint_dim: int = 1024
    num_heads: int = 32
    text_dim: int = 512
    use_shared_ffn: bool = True
    use_cls_only: bool = False

    # Knowledge graph parameters
    kg_method: str = "cosine"
    kg_emb_dim: int = 200
    kg_epochs: int = 5
    kg_weight: float = 0.1
    kg_mode: str = "dataset"

    # Loss parameters
    cls_weight: float = 1.5
    cont_weight: float = 0.3
    weight_img_joint: float = 0.1
    weight_text_joint: float = 0.1
    gamma_focal: float = 1.0
    focal_ratio: float = 0.3
    temperature: float = 0.125

    # Wandb parameters
    project_name: str = "multi-modal-retrieval-predict"

    # Auto-generated
    run_name: str = field(init=False, default="")

    def __post_init__(self):
        if self.use_hybrid:
            name_method = "hybrid(bce_focal)"
        elif self.use_focal:
            name_method = "focal"
        else:
            name_method = "bce"

        self.set_run_name(name_method)

    def set_run_name(self, name_method: str):
        # skip run_name when building dict
        cfg_dict = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "run_name"}

        parts = [f"method={name_method}"]
        for k, v in cfg_dict.items():
            if isinstance(v, float):
                if "lr" in k:
                    parts.append(f"{k}={v:.0e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.run_name = "_".join(parts)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            yaml_data = yaml.safe_load(f)

        used_defaults = []
        data = {}
        for f in fields(Config):
            if not f.init:
                continue
            if f.name in yaml_data:
                data[f.name] = yaml_data[f.name]
            else:
                data[f.name] = f.default
                used_defaults.append(f.name)

        cfg = Config(**data)
        if used_defaults:
            print("Using default values for:", ", ".join(used_defaults))
        return cfg
