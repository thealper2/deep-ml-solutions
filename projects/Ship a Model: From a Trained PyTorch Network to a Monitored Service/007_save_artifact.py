import io
import torch


def save_artifact(model, metadata):
    """Bytes holding the state dict, the hidden width and the metadata."""
    buffer = io.BytesIO()
    payload = {
        "state_dict": model.state_dict(),
        "hidden": model.layers[0].out_features,
        "metadata": dict(metadata),
    }
    torch.save(payload, buffer)
    return buffer.getvalue()