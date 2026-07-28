from tinygrad import Tensor, nn
import tinygrad.nn as tnn

def build_model():
    class TinyNet:
        def __init__(self):
            self.conv1 = tnn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = tnn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.conv3 = tnn.Conv2d(8, 16, 1)
            self.fc = tnn.Linear(16, 10)
            
        def __call__(self, x: Tensor) -> Tensor:
            x = x.relu()
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = x.mean(axis=(2, 3))
            x = self.fc(x)
            return x
    
    return TinyNet()
