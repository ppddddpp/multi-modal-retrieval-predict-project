from dataclasses import dataclass, field
import yaml

@dataclass
class Config:
    epochs: int = 50
    patience: int = 10
    batch_size: int = 1
    lr: float = 2e-5
    use_focal: bool = False
    use_hybrid: bool = True
    image_backbone: str = "swin"
    fusion_type: str = "cross"
    joint_dim: int = 1024
    gamma_focal: float = 1.0
    focal_ratio: float = 0.3
    text_dim: int = 512
    temperature: float = 0.125
    kg_mode: str = "dataset"
    kg_method: str = "cosine"
    kg_emb_dim: int = 200
    kg_epochs: int = 5 
    kg_weight: float = 0.1
    cls_weight: float = 1.5
    cont_weight: float = 0.3
    num_heads: int = 32
    num_fusion_layers: int = 3
    use_shared_ffn: bool = True
    use_cls_only: bool = False
    project_name: str = "multi-modal-retrieval-predict"
    # this will be auto‑generated
    run_name: str = field(init=False)

    def __post_init__(self):
        name_method = ""
        if self.use_hybrid:
            name_method = "hybrid(bce_focal)"
        elif self.use_focal:
            name_method = "focal"
        else:
            name_method = "bce"
        
        self.run_name = (
            f"method={name_method}"
            f"_ratio={self.focal_ratio}"
            f"_backbone={self.image_backbone}"
            f"_fusion={self.fusion_type}"
            f"_eps={self.epochs}"
            f"_b={self.batch_size}"
            f"_lr={self.lr:.0e}"
            f"_jd={self.joint_dim}"
            f"_nh={self.num_heads}"
            f"_txt_dim={self.text_dim}"
            f"_temp={self.temperature}"
            f"_kg_mode={self.kg_mode}"
            f"_kg_method={self.kg_method}"
            f"_kg_emb={self.kg_emb_dim}"
            f"_kg_epochs={self.kg_epochs}"
            f"_kg_weight={self.kg_weight}"
            f"_cls={self.cls_weight}"
            f"_cont={self.cont_weight}"
            f"_gamma={self.gamma_focal}"
            f"_fusion_layers={self.num_fusion_layers}"
            f"_shared_ffn={self.use_shared_ffn}"
            f"_cls_only={self.use_cls_only}"
        )

    @staticmethod
    def load(path: str) -> "Config":
        """
        Load a Config from a yaml file.

        Args:
            path (str): Path to a yaml file containing the configuration.

        Returns:
            Config: A Config object loaded from the yaml file.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
            
        data["epochs"] = int(data.get("epochs", 50))
        data["patience"] = int(data.get("patience", 10))
        data["batch_size"] = int(data.get("batch_size", 1))
        data["lr"] = float(data.get("lr", 2e-5))
        data["gamma_focal"] = float(data.get("gamma_focal", 1.0))
        data["focal_ratio"] = float(data.get("focal_ratio", 0.3))
        data["temperature"] = float(data.get("temperature", 0.125))
        data["kg_method"] = str(data.get("kg_method", "cosine"))
        data["kg_mode"] = str(data.get("kg_mode", "dataset"))
        data["kg_emb_dim"] = int(data.get("kg_emb_dim", 200))
        data["kg_epochs"] = int(data.get("kg_epochs", 5))
        data["kg_weight"] = float(data.get("kg_weight", 0.1))
        data["cls_weight"] = float(data.get("cls_weight", 1.5))
        data["cont_weight"] = float(data.get("cont_weight", 0.3))
        data["num_heads"] = int(data.get("num_heads", 32))
        data["text_dim"] = int(data.get("text_dim", 512))
        data["use_focal"] = bool(data.get("use_focal", False))
        data["use_hybrid"] = bool(data.get("use_hybrid", True))
        data["fusion_type"] = str(data.get("fusion_type", "cross"))
        data["joint_dim"] = int(data.get("joint_dim", 1024))
        data["project_name"] = str(data.get("project_name", "multi-modal-retrieval-predict"))
        data["num_fusion_layers"] = int(data.get("num_fusion_layers", 3))
        data["use_shared_ffn"] = bool(data.get("use_shared_ffn", True))
        data["use_cls_only"] = bool(data.get("use_cls_only", False))

        return Config(**data)
