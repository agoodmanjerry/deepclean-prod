
import torch.nn as nn

class DeepClean(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(in_channels),
            nn.Tanh()
        )
        
        self.downsampler = nn.Sequential()
        self.downsampler.add_module('CONV_1', nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_1', nn.BatchNorm1d(8))
        self.downsampler.add_module('TANH_1', nn.Tanh())
        self.downsampler.add_module('CONV_2', nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_2', nn.BatchNorm1d(16))
        self.downsampler.add_module('TANH_2', nn.Tanh())
        self.downsampler.add_module('CONV_3', nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_3', nn.BatchNorm1d(32))
        self.downsampler.add_module('TANH_3', nn.Tanh())
        self.downsampler.add_module('CONV_4', nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_4', nn.BatchNorm1d(64))
        self.downsampler.add_module('TANH_4', nn.Tanh())
                                      
        self.upsampler = nn.Sequential()
        self.upsampler.add_module(
            'CONVTRANS_1', nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_1', nn.BatchNorm1d(32))
        self.upsampler.add_module('TANH_1', nn.Tanh())
        self.upsampler.add_module(
            'CONVTRANS_2', nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_2', nn.BatchNorm1d(16))
        self.upsampler.add_module('TANH_2', nn.Tanh())
        self.upsampler.add_module(
            'CONVTRANS_3', nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_3', nn.BatchNorm1d(8))
        self.upsampler.add_module('TANH_3', nn.Tanh())
        self.upsampler.add_module(
            'CONVTRANS_4', nn.ConvTranspose1d(8,in_channels, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_4', nn.BatchNorm1d(in_channels))
        self.upsampler.add_module('TANH_4', nn.Tanh())
        
        self.output_conv = nn.Conv1d(in_channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.downsampler(x)
        x = self.upsampler(x)
        x = self.output_conv(x)
        x = x.view(x.shape[0], -1)
        return x