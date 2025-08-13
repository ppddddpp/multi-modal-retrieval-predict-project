try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image

def compare_maps(map_a: np.ndarray, map_b: np.ndarray, topk_frac: float = 0.05):
    """
    Compute metrics comparing two attention maps.

    Parameters
    ----------
    map_a : np.ndarray
        First attention map.
    map_b : np.ndarray
        Second attention map.
    topk_frac : float, optional
        Fraction of top scoring pixels to use for IoU calculation. Defaults to 0.05.

    Returns
    -------
    dict
        Dictionary containing the following metrics:
        - 'pearson': Pearson correlation coefficient between flattened maps.
        - 'spearman': Spearman rank correlation coefficient between flattened maps.
        - f'iou_top{int(topk_frac*100)}pct': IoU on top-k fraction of pixels.
    """
    a = map_a.flatten().astype(np.float32)
    b = map_b.flatten().astype(np.float32)
    # normalize
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)

    # pearson
    if _HAVE_SCIPY:
        pear = float(pearsonr(a, b)[0])
        spear = float(spearmanr(a, b).correlation)
    else:
        # fallback pearson via numpy
        if np.std(a) == 0 or np.std(b) == 0:
            pear = 0.0
        else:
            pear = float(np.corrcoef(a, b)[0, 1])
        # approximate spearman by ranking then pearson
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        if np.std(ar) == 0 or np.std(br) == 0:
            spear = 0.0
        else:
            spear = float(np.corrcoef(ar, br)[0, 1])

    # IoU on top-k fraction
    k = max(1, int(len(a) * topk_frac))
    a_top = (a >= np.partition(a, -k)[-k])
    b_top = (b >= np.partition(b, -k)[-k])
    inter = np.logical_and(a_top, b_top).sum()
    union = np.logical_or(a_top, b_top).sum()
    iou = float(inter) / (float(union) + 1e-8)

    return {'pearson': pear, 'spearman': spear, f'iou_top{int(topk_frac*100)}pct': iou}

def heatmap_to_base64_overlay(orig_img: np.ndarray,
                            heatmap: np.ndarray,
                            cmap: str = 'jet',
                            alpha: float = 0.5) -> str:
    # convert orig to 2D grayscale if needed
    """
    Returns a base64-encoded PNG of the original image with the heatmap overlayed.

    Parameters
    ----------
    orig_img : np.ndarray
        The original image. May be 2D (grayscale) or 3D (RGB).
    heatmap : np.ndarray
        The heatmap. Will be resized if shape is different from `orig_img`.
    cmap : str, optional
        The matplotlib colormap to use. Defaults to 'jet'.
    alpha : float, optional
        The alpha channel value of the heatmap. Defaults to 0.5.

    Returns
    -------
    str
        The base64-encoded PNG.
    """
    img = orig_img.squeeze()
    if img.ndim == 3 and img.shape[2] == 3:
        base = img
    else:
        # grayscale -> RGB
        if img.max() <= 1.0:
            img_u8 = (img * 255).astype('uint8')
        else:
            img_u8 = img.astype('uint8')
        base = np.stack([img_u8, img_u8, img_u8], axis=-1)
        base = base.astype('uint8')

    # normalize heatmap to 0..1
    h = heatmap.astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    # resize heatmap if different shape
    H, W = base.shape[:2]
    if h.shape != (H, W):
        h_img = Image.fromarray((h * 255).astype('uint8')).resize((W, H), resample=Image.BILINEAR)
        h = np.asarray(h_img).astype(np.float32) / 255.0

    # colorize via matplotlib cmap
    cmap_fn = plt.get_cmap(cmap)
    colored = cmap_fn(h)[:, :, :3]  # HxWx3 float 0..1
    colored_u8 = (colored * 255).astype('uint8')

    # ensure base is uint8
    if base.dtype != np.uint8:
        if base.max() <= 1.0:
            base_u8 = (base * 255).astype('uint8')
        else:
            base_u8 = base.astype('uint8')
    else:
        base_u8 = base

    # blend
    blended = (base_u8 * (1 - alpha) + colored_u8 * alpha).astype('uint8')

    pil = Image.fromarray(blended)
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')
