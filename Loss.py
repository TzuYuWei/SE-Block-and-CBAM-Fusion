# TVLoss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=0.01):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
   
# **VGG 特徵提取器 (支持多層特徵)**
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=[3, 8, 17, 26]):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features  # 修改這一行
        # 根據層索引生成對應的切片
        self.slices = nn.ModuleList([nn.Sequential(*[vgg[i] for i in range(start, end)])
                                     for start, end in zip([0] + layers[:-1], layers)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features

def frequency_loss(real_img, fake_img, weight=None, eps=1e-8):
    # 計算 FFT
    real_fft = torch.fft.fft2(real_img, norm='ortho')
    fake_fft = torch.fft.fft2(fake_img, norm='ortho')

    # 拆成實部與虛部
    real_real = real_fft.real
    real_imag = real_fft.imag
    fake_real = fake_fft.real
    fake_imag = fake_fft.imag

    # 計算 L1 loss（可以選擇加權 focal mask）
    if weight is not None:
        loss_real = F.l1_loss(real_real * weight, fake_real * weight)
        loss_imag = F.l1_loss(real_imag * weight, fake_imag * weight)
    else:
        loss_real = F.l1_loss(real_real, fake_real)
        loss_imag = F.l1_loss(real_imag, fake_imag)

    loss = loss_real + loss_imag
    return loss

def align_images(*images):
    min_height = min(img.size(2) for img in images)
    min_width = min(img.size(3) for img in images)
    return [img[:, :, :min_height, :min_width] for img in images]