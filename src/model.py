import torch
import torch.nn as nn
from fusion import CrossModalFusion, Backbones

class MultiModalRetrievalModel(nn.Module):
    """
    Wraps Backbones + Fusion + Classification head
    into one model that produces:
      - joint_emb (B, joint_dim)
      - logits    (B, num_classes)
    """
    def __init__(
        self,
        joint_dim:    int = 256,
        num_heads:    int = 4,
        num_classes:  int = 22,
        fusion_type:  str = "cross",
        swin_name:    str = "swin_base_patch4_window7_224",
        bert_name:    str = "emilyalsentzer/Bio_ClinicalBERT",
        swin_ckpt_path:    str = None,
        bert_local_dir:   str = None,
        pretrained:   bool = True,
    ):
        """
        :param joint_dim: dimensionality of the joint embedding
        :param num_heads: number of attention heads for CrossModalFusion
        :param num_classes: number of output classes
        :param fusion_type: type of fusion module to use; one of "cross", "simple", "gated"
        :param swin_name: name of the Swin transformer model to use
        :param bert_name: name of the ClinicalBERT model to use
        :param swin_ckpt_path: path to a Swin transformer checkpoint to load
        :param bert_local_dir: directory containing a ClinicalBERT model to load
        :param pretrained: whether to load pre-trained weights for the Swin and ClinicalBERT models

        Args:
            joint_dim (int, optional): dimensionality of the joint embedding. Defaults to 256.
            num_heads (int, optional): number of attention heads for CrossModalFusion. Defaults to 4.
            num_classes (int, optional): number of output classes. Defaults to 14.
            fusion_type (str, optional): type of fusion module to use; one of "cross", "simple", "gated". Defaults to "cross".
            swin_name (str, optional): name of the Swin transformer model to use. Defaults to "swin_base_patch4_window7_224".
            bert_name (str, optional): name of the ClinicalBERT model to use. Defaults to "emilyalsentzer/Bio_ClinicalBERT".
            swin_ckpt_path (str, optional): path to a Swin transformer checkpoint to load. Defaults to None.
            bert_local_dir (str, optional): directory containing a ClinicalBERT model to load. Defaults to None.
            pretrained (bool, optional): whether to load pre-trained weights for the Swin and ClinicalBERT models. Defaults to True.
        """
        super().__init__()

        # instantiate vision+text backbones
        self.backbones = Backbones(
            swin_model_name    = swin_name,
            bert_model_name    = bert_name,
            swin_checkpoint_path = swin_ckpt_path,
            bert_local_dir       = bert_local_dir,
            pretrained           = pretrained
        )
        img_dim = self.backbones.img_dim
        txt_dim = self.backbones.txt_dim

        # set up fusion
        if fusion_type == "cross":
            self.fusion = CrossModalFusion(img_dim, txt_dim, joint_dim, num_heads)
        else:
            raise ValueError(f"Unknown fusion_type {fusion_type!r}")

        # classification head on the joint embedding
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, joint_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(joint_dim//2, num_classes)
        )

    def forward(self, image, input_ids, attention_mask, return_attention=False):
        """
        Inputs:
          pixel_values  (B, 1, H, W) or (B, 3, H, W)
          input_ids     (B, L)
          attention_mask(B, L)
        Returns:
          joint_emb (B, joint_dim)
          logits    (B, num_classes)
        """
        # extracting separate features
        (img_global, img_patch), txt_feat = self.backbones(image, input_ids, attention_mask)

        # fuse into shared embedding
        if return_attention:
            joint_emb, attn_weights = self.fusion(img_global, img_patch, txt_feat, return_attention=True)
        else:
            joint_emb = self.fusion(img_global, img_patch, txt_feat)
            attn_weights = None
        logits = self.classifier(joint_emb)
        
        return joint_emb, logits, attn_weights
