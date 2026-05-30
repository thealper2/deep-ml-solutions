from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d

class ResidualBlock:
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.uses_projection = not (in_channels == out_channels and stride == 1)

        self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)

        self.conv2 = Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)

        if self.uses_projection:
            self.shortcut_conv = Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.shortcut_bn = BatchNorm2d(out_channels)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def __call__(self, x: Tensor) -> Tensor:
        out = self.conv1(x).relu()
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.uses_projection:
            skip = self.shortcut_conv(x)
            skip = self.shortcut_bn(skip)
        else:
            skip = x

        return (out + skip).relu()