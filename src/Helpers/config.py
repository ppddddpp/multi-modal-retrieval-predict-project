from dataclasses import dataclass, field, fields, asdict
import yaml
from typing import Any, Dict

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
        # run basic validation so bad values are caught early
        self.validate()

    def set_run_name(self, name_method: str):
        # build dict skipping run_name, only include init fields
        cfg_dict = {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.init and fld.name != "run_name"
        }

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

    def validate(self) -> None:
        """Basic sanity checks raise ValueError if something invalid."""
        if not (isinstance(self.lr, (float, int)) and self.lr >= 0.0):
            raise ValueError(f"lr must be a non-negative number. Got {self.lr!r}")
        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            raise ValueError(f"batch_size must be a positive integer. Got {self.batch_size!r}")
        if not (isinstance(self.epochs, int) and self.epochs > 0):
            raise ValueError(f"epochs must be a positive integer. Got {self.epochs!r}")
        if not (isinstance(self.temperature, (float, int)) and self.temperature > 0.0):
            raise ValueError(f"temperature must be > 0. Got {self.temperature!r}")

    @staticmethod
    def _coerce_value(raw: Any, target_type: Any) -> Any:
        """Try to coerce common scalar types. If coercion fails, return raw."""
        if raw is None:
            return None
        try:
            if target_type is float:
                return float(raw)
            if target_type is int:
                return int(raw)
            if target_type is bool:
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, str):
                    return raw.strip().lower() in ("true", "1", "yes", "y")
                return bool(raw)
            if target_type is str:
                return str(raw)
        except Exception:
            # fallback to raw value if conversion fails
            return raw
        return raw

    @staticmethod
    def load(path: str) -> "Config":
        """Load config from YAML, coerce types for dataclass fields and validate."""
        with open(path, "r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

        if not isinstance(yaml_data, dict):
            raise ValueError("Config file must contain a YAML mapping (keys/values).")

        # map of init-able dataclass fields
        field_map: Dict[str, Any] = {fld.name: fld for fld in fields(Config) if fld.init}

        # warn about unknown keys in YAML
        extra_keys = set(yaml_data.keys()) - set(field_map.keys())
        if extra_keys:
            print(f"[WARN] Unknown keys in config file (will be ignored): {sorted(extra_keys)}")

        used_defaults = []
        data: Dict[str, Any] = {}
        for name, fld in field_map.items():
            if name in yaml_data:
                raw = yaml_data[name]
                coerced = Config._coerce_value(raw, fld.type)
                data[name] = coerced
            else:
                data[name] = fld.default
                used_defaults.append(name)

        cfg = Config(**data)  # __post_init__ will run and call validate()
        if used_defaults:
            print("Using default values for:", ", ".join(used_defaults))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Return plain dict representation (includes run_name)."""
        return asdict(self)

if __name__ == "__main__":
    import os
    example_path = os.path.join("D:\Github\multi-modal-retrieval-predict-project\configs\config.yaml")
    if os.path.exists(example_path):
        cfg = Config.load(example_path)
        print("Loaded config:", cfg)
        print("run_name:", cfg.run_name)
        print("LR type:", type(cfg.lr), "value:", cfg.lr)
    else:
        # fallback: show default config
        cfg = Config()
        print("No config.yaml found — using defaults:")
        print(cfg)
        print("run_name:", cfg.run_name)
