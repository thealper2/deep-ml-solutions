from typing import Optional, Union

try:
    import numpy as np
except Exception:
    np = None

ArrayLike = Union[list, "np.ndarray"]

def flow_epe(pred: ArrayLike,
             gt: ArrayLike,
             mask: Optional[ArrayLike] = None,
             max_flow: Optional[float] = None) -> float:
    """
    Compute mean End-Point Error (EPE) between predicted and ground-truth optical flow.

    Args:
        pred, gt: (H, W, 2) lists or NumPy arrays.
        mask: optional (H, W) or broadcastable to (H, W); 1=include, 0=ignore.
        max_flow: optional float; clip per-pixel EPE to this value.

    Returns:
        float: mean EPE over valid pixels. Returns -1 on invalid input or if no valid pixels.
    """
    try:
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        if not isinstance(gt, np.ndarray):
            gt = np.array(gt)
        if mask is not None and not isinstance(mask, np.ndarray):
            mask = np.array(mask)
    except Exception:
        return -1.0
    
    if pred.shape != gt.shape or len(pred.shape) != 3 or pred.shape[-1] != 2:
        return -1.0
    
    diff = pred - gt
    epe = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    valid = np.isfinite(epe)
    
    if mask is not None:
        if mask.shape != epe.shape and mask.shape != (epe.shape[0], epe.shape[1]):
            return -1.0
        valid = valid & (np.array(mask, dtype=bool))
    
    valid_epe = epe[valid]
    
    if len(valid_epe) == 0:
        return -1.0
    
    if max_flow is not None:
        valid_epe = np.clip(valid_epe, None, max_flow)
    
    mean_epe = float(np.mean(valid_epe))
    return mean_epe