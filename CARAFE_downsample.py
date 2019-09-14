import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F


class CarafeDownsample(nn.Module):
    def __init__(self, in_channels, scale_factor, m_channels=64,
                 encoder_kernel_size=3, downsample_kernel_size=5):
        super(CarafeDownsample, self).__init__()
        assert(isinstance(scale_factor, numbers.Integral))
        assert(downsample_kernel_size - encoder_kernel_size == 2)
        self.compress_conv = nn.Conv2d(in_channels, m_channels, 1, 1, 0)
        self.encoder_conv = nn.Conv2d(m_channels, downsample_kernel_size**2, encoder_kernel_size, scale_factor, encoder_kernel_size//2)
        self.down_k = downsample_kernel_size
        self.unfold = nn.Unfold(kernel_size=self.down_k, stride=scale_factor, padding=self.down_k//2)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.down_k
        nH = H // self.scale_factor
        nW = W // self.scale_factor
        kernel = self.compress_conv(x)
        kernel = self.encoder_conv(kernel)
        kernel = F.softmax(kernel, dim=1)
        x = self.unfold(x).view(B, C, k*k, nH, nW)
        kernel = kernel.unsqueeze(1)
        x = x * kernel
        x = x.sum(dim=2)
        return x


if __name__ == "__main__":
    inputs = torch.randn(4, 256, 32, 28)
    model = CarafeDownsample(in_channels=256, scale_factor=2)
    output = model(inputs)
    print(output.shape)
