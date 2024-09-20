import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithAttentionAnd3DConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):  # Default out_channels to 1
        super(UNetWithAttentionAnd3DConv, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64, is_3d=True)
        self.encoder2 = self.conv_block(64, 128, is_3d=True)
        self.encoder3 = self.conv_block(128, 256, is_3d=True)
        self.encoder4 = self.conv_block(256, 512, is_3d=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        self.bottleneck = self.conv_block(512, 1024, is_3d=True)

        self.attention = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, kernel_size=1),
            nn.Sigmoid()
        )

        self.decoder4 = self.conv_block(1024 + 512, 512, is_3d=True)
        self.decoder3 = self.conv_block(512 + 256, 256, is_3d=True)
        self.decoder2 = self.conv_block(256 + 128, 128, is_3d=True)
        self.decoder1 = self.conv_block(128 + 64, 64, is_3d=True)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, is_3d=False):
        if is_3d:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        x = x.reshape(batch_size, channels, seq_len, height, width)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        attention = self.attention(bottleneck)
        bottleneck = bottleneck * attention

        dec4 = self.decoder4(torch.cat([F.adaptive_avg_pool3d(bottleneck, output_size=enc4.shape[2:]), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.adaptive_avg_pool3d(dec4, output_size=enc3.shape[2:]), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.adaptive_avg_pool3d(dec3, output_size=enc2.shape[2:]), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.adaptive_avg_pool3d(dec2, output_size=enc1.shape[2:]), enc1], dim=1))

        return self.final_conv(F.adaptive_avg_pool3d(dec1, output_size=(seq_len, height, width)))
