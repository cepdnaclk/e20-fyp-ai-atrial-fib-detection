"""
Shared model architectures for Phase 3 Counterfactual Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')


class Encoder(nn.Module):
    """High-Quality VAE Encoder"""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 15, 2, 7), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 11, 2, 5), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 7, 2, 3), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 5, 2, 2), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 3, 2, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc_mu = nn.Linear(512 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16, latent_dim)
        
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """High-Quality Decoder"""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 16), nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 4, 2, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, 4, 2, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
        )
        self.final = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 3, 1, 1),
        )
        
    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 16)
        h = self.deconv(h)
        h = F.interpolate(h, size=2500, mode='linear', align_corners=False)
        return self.final(h)


class StyleModifier(nn.Module):
    """Learns to modify latent code to flip class while minimizing change."""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim),
        )
        self.net[-1].weight.data *= 0.01
        self.net[-1].bias.data.zero_()
        
    def forward(self, z, target_class):
        inp = torch.cat([z, target_class.float()], dim=1)
        delta = self.net(inp)
        return z + delta


class CounterfactualVAE(nn.Module):
    """Complete model for counterfactual generation."""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.style_modifier = StyleModifier(latent_dim)
        self.latent_dim = latent_dim
        
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def generate_counterfactual(self, x, target_class):
        z, _, _ = self.encode(x)
        z_modified = self.style_modifier(z, target_class)
        return self.decode(z_modified), z, z_modified


class ClassifierWrapper(nn.Module):
    """Wrapper that handles LSTM backward pass requirements."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        
    def forward(self, x):
        self.model.train()
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        logits, _ = self.model(x_norm)
        return logits


def load_classifier(device):
    """Load the pre-trained AFibResLSTM classifier."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / 'models'))
    from model_architecture import AFibResLSTM, ModelConfig
    
    classifier_path = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(device)
    checkpoint = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    return classifier
