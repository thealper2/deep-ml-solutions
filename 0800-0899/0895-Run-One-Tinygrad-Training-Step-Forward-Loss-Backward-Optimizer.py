from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import SGD

def train_one_step(model, x: Tensor, y: Tensor, lr: float) -> float:
    with Tensor.train():
        params = nn.state.get_parameters(model)
        optimizer = SGD(params, lr=lr)
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()