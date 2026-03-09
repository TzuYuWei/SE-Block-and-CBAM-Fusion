# **Spectral Normalization **
class SpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SpectralNormConv2d, self).__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.conv(x)
# **多尺度 PatchGAN 判別器**
class MultiScalePatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=3, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SpectralPatchGANDiscriminator(in_channels, base_channels, n_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(d(x))
        return outputs

# ** PatchGAN 判別器**
class SpectralPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SpectralPatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            SpectralNormConv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)