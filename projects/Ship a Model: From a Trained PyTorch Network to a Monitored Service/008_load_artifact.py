import io
import torch


def load_artifact(blob):
    """Rebuild a DeliveryNet from save_artifact bytes; return (model in eval mode, metadata)."""
    buffer = io.BytesIO(blob)
    payload = torch.load(buffer, weights_only=True)
    state_dict = payload["state_dict"]
    hidden = payload["hidden"]
    metadata = payload["metadata"]

    mean = state_dict["mean"].numpy()
    std = state_dict["std"].numpy()
    model = DeliveryNet(mean, std, hidden=hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata