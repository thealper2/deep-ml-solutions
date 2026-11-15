import io
import torch


def export_model(model, example_input):
    """Trace the model in eval mode and return TorchScript bytes."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    return buffer.getvalue()
