import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F


class CarafeUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor, m_channels=64,
                 encoder_kernel_size=3, upsample_kernel_size=5):
        super(CarafeUpsample, self).__init__()
        assert(isinstance(scale_factor, numbers.Integral))
        assert(upsample_kernel_size - encoder_kernel_size == 2)
        self.compress_conv = nn.Conv2d(in_channels, m_channels, 1, 1, 0)
        self.encoder_conv = nn.Conv2d(m_channels, upsample_kernel_size**2 * scale_factor**2, encoder_kernel_size, 1, encoder_kernel_size//2)
        self.pixelshuffle = nn.PixelShuffle(scale_factor)
        self.up_k = upsample_kernel_size
        self.unfold = nn.Unfold(kernel_size=self.up_k, stride=1, padding=self.up_k//2)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.up_k
        nH = self.scale_factor*H
        nW = self.scale_factor*W
        kernel = self.compress_conv(x)
        kernel = self.encoder_conv(kernel)
        kernel = self.pixelshuffle(kernel)
        kernel = F.softmax(kernel, dim=1)
        x = self.unfold(x).view(B, -1, H, W)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = x.view(B, C, k*k, nH, nW)
        kernel = kernel.unsqueeze(1)
        x = x * kernel
        x = x.sum(dim=2)
        return x


if __name__ == "__main__":
    inputs = torch.randn(4, 256, 32, 28)
    model = CarafeUpsample(in_channels=256, scale_factor=2)
    output = model(inputs)
    print(output.shape)
