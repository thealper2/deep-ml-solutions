from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

def accumulated_step(model, micro_batches, optimizer, criterion):
    Tensor.training = True

    for p in get_parameters(model):
        p.grad = None

    num_batches = len(micro_batches)
    total_loss = 0.0

    for x, y in micro_batches:
        loss = criterion(model(x), y)
        (loss / num_batches).backward()
        total_loss += loss.item()

    optimizer.step()

    return float(total_loss / num_batches)