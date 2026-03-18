"""
Image-to-Image Diffusion for ECG Counterfactual Generation
============================================================

Core Idea: SDEdit-style approach where we:
1. Add partial noise to the original ECG (controlled by noise_strength)
2. Denoise with target class conditioning
3. This preserves patient morphology while editing class-relevant features

Key Advantages over Pure Diffusion:
- Morphology preservation: We don't start from random noise
- Controllable edit magnitude via noise_strength parameter
- Faster inference: Fewer denoising steps needed

Architecture:
- UNet1DConditional: Denoising network with AdaGN conditioning
- Class Embedding: Learns class-specific features for Normal/AFib
- Content Encoder: Optional - extracts morphological features to preserve
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# ADAPTIVE GROUP NORMALIZATION
# ============================================================================

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization for conditioning injection.
    Uses condition vector to predict scale/shift for feature modulation.
    """
    def __init__(self, num_channels, cond_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)  # Scale + Shift
        self.act = nn.SiLU()

    def forward(self, x, cond):
        """
        Args:
            x: [batch, channels, length] - Feature map
            cond: [batch, cond_dim] - Conditioning vector
        """
        params = self.proj(cond)
        scale, shift = params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        h = self.norm(x)
        h = h * (1 + scale) + shift
        return h


# ============================================================================
# UNET1D FOR IMAGE-TO-IMAGE DIFFUSION
# ============================================================================

class UNet1DImg2Img(nn.Module):
    """
    UNet for ECG Image-to-Image Diffusion.
    
    Differences from standard diffusion UNet:
    - Takes noisy original as input (not pure noise)
    - Class conditioning is the main control signal
    - Optional content conditioning for stronger morphology preservation
    """
    
    def __init__(self, in_channels=1, out_channels=1, 
                 base_channels=64, channel_mults=(1, 2, 4, 8),
                 time_embed_dim=256, num_classes=2, class_embed_dim=256,
                 dropout=0.1, use_content_conditioning=False, content_dim=512):
        super().__init__()
        
        self.use_content_conditioning = use_content_conditioning
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(class_embed_dim, class_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(class_embed_dim * 2, time_embed_dim)
        )
        
        # Optional content conditioning
        if use_content_conditioning:
            self.content_proj = nn.Sequential(
                nn.Linear(content_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
        
        # Combined conditioning dimension
        cond_dim = time_embed_dim
        
        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        channels = [base_channels * m for m in channel_mults]
        in_ch = in_channels
        
        for i, out_ch in enumerate(channels):
            self.encoder_blocks.append(
                ResBlock1D(in_ch, out_ch, cond_dim, dropout=dropout)
            )
            self.encoder_pools.append(nn.MaxPool1d(2) if i < len(channels) - 1 else nn.Identity())
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = ResBlock1D(channels[-1], channels[-1], cond_dim, dropout=dropout)
        
        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i] + rev_channels[i + 1]  # Skip connection
            out_ch = rev_channels[i + 1]
            self.decoder_blocks.append(
                ResBlock1D(in_ch, out_ch, cond_dim, dropout=dropout)
            )
            self.upsample.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], out_channels, kernel_size=1)
        )
        
    def forward(self, x, timestep, class_label, content_emb=None):
        """
        Args:
            x: [batch, 1, 2500] - Noisy ECG
            timestep: [batch] - Noise timestep
            class_label: [batch] - Target class (0=Normal, 1=AFib)
            content_emb: [batch, content_dim] - Optional content embedding
            
        Returns:
            noise_pred: [batch, 1, 2500] - Predicted noise
        """
        # Time embedding
        time_emb = self.time_mlp(timestep)
        
        # Class embedding
        class_emb = self.class_embed(class_label)
        class_emb = self.class_proj(class_emb)
        
        # Combine embeddings
        cond = time_emb + class_emb
        
        if self.use_content_conditioning and content_emb is not None:
            content_emb = self.content_proj(content_emb)
            cond = cond + content_emb
        
        # Encoder
        skip_connections = []
        h = x
        for block, pool in zip(self.encoder_blocks, self.encoder_pools):
            h = block(h, cond)
            skip_connections.append(h)
            h = pool(h)
        
        # Bottleneck
        h = self.bottleneck(h, cond)
        
        # Decoder
        for i, (block, up) in enumerate(zip(self.decoder_blocks, self.upsample)):
            h = up(h)
            # Handle size mismatch
            skip = skip_connections[-(i + 2)]
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=True)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)
        
        # Output
        # First concat with first skip connection if sizes match
        if h.shape[-1] != skip_connections[0].shape[-1]:
            h = F.interpolate(h, size=skip_connections[0].shape[-1], mode='linear', align_corners=True)
        
        return self.out_conv(h)


class ResBlock1D(nn.Module):
    """Residual block with AdaGN conditioning"""
    
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = AdaGN(out_ch, cond_dim)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = AdaGN(out_ch, cond_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # Skip connection
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        
    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = self.act(h)
        h = self.dropout(h)
        
        h = self.conv2(h)
        h = self.norm2(h, cond)
        h = self.act(h)
        
        return h + self.skip(x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================================
# CONTENT ENCODER (OPTIONAL)
# ============================================================================

class ContentEncoder(nn.Module):
    """
    Extracts patient-specific morphological features.
    Trained to be class-invariant (same features for Normal/AFib versions of same patient)
    """
    
    def __init__(self, input_length=2500, hidden_dim=256, output_dim=512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 625
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 156
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 39
            
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, 2500] - ECG signal
        Returns:
            content: [batch, output_dim] - Morphological embedding
        """
        h = self.encoder(x)
        h = h.squeeze(-1)
        return self.fc(h)


# ============================================================================
# FULL IMAGE-TO-IMAGE DIFFUSION MODEL
# ============================================================================

class ECGImg2ImgDiffusion(nn.Module):
    """
    Complete Image-to-Image Diffusion model for ECG counterfactuals.
    
    Training:
        1. Original ECG -> Add noise at random timestep
        2. Noisy ECG + Target class -> UNet predicts noise
        3. Loss = ||predicted_noise - true_noise||
        
    Inference (SDEdit-style):
        1. Original ECG -> Add noise at strength-controlled timestep
        2. Denoise with target class conditioning
        3. Output: Counterfactual that preserves morphology
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Main UNet
        self.unet = UNet1DImg2Img(
            in_channels=1,
            out_channels=1,
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            time_embed_dim=config['time_embed_dim'],
            num_classes=2,
            class_embed_dim=config['class_embed_dim'],
            dropout=config.get('dropout', 0.1),
            use_content_conditioning=config.get('use_content_conditioning', False),
            content_dim=config.get('content_dim', 512)
        )
        
        # Optional content encoder
        if config.get('use_content_conditioning', False):
            self.content_encoder = ContentEncoder(
                output_dim=config.get('content_dim', 512)
            )
        else:
            self.content_encoder = None
            
    def forward(self, x, timestep, class_label):
        """Forward pass for training"""
        content_emb = None
        if self.content_encoder is not None:
            with torch.no_grad():
                content_emb = self.content_encoder(x)
        
        return self.unet(x, timestep, class_label, content_emb)
    
    def get_content_embedding(self, x):
        """Extract content embedding for morphology preservation"""
        if self.content_encoder is not None:
            return self.content_encoder(x)
        return None


# ============================================================================
# NOISE SCHEDULER WRAPPER
# ============================================================================

class Img2ImgNoiseScheduler:
    """
    Wrapper for DDIM scheduler with image-to-image specific functions.
    
    Key method: add_noise_at_strength()
    - Instead of random timestep, uses strength (0-1) to control how much noise
    - strength=0: No noise, output = input
    - strength=1: Full noise, output = pure Gaussian noise
    """
    
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 beta_schedule='linear'):
        from diffusers import DDIMScheduler
        
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type='epsilon',
            clip_sample=False
        )
        
        self.num_train_timesteps = num_train_timesteps
        
    def add_noise(self, x, noise, timestep):
        """Standard noise addition for training"""
        return self.scheduler.add_noise(x, noise, timestep)
    
    def add_noise_at_strength(self, x, noise, strength):
        """
        Add noise controlled by strength (0-1).
        
        Args:
            x: [batch, 1, 2500] - Original ECG
            noise: [batch, 1, 2500] - Gaussian noise
            strength: float in [0, 1] - How much noise to add
            
        Returns:
            noisy_x: Noisy version of x
            timestep: Corresponding timestep for this strength
        """
        # Map strength to timestep
        timestep = int(strength * (self.num_train_timesteps - 1))
        timestep = torch.tensor([timestep], device=x.device).long()
        
        return self.scheduler.add_noise(x, noise, timestep), timestep
    
    def step(self, noise_pred, timestep, sample):
        """Single denoising step"""
        return self.scheduler.step(noise_pred, timestep, sample).prev_sample
    
    def set_timesteps(self, num_inference_steps):
        """Set inference timesteps"""
        self.scheduler.set_timesteps(num_inference_steps)
        
    @property
    def timesteps(self):
        return self.scheduler.timesteps


# ============================================================================
# DEFAULT CONFIG
# ============================================================================

def get_default_config():
    return {
        'base_channels': 64,
        'channel_mults': (1, 2, 4, 8),
        'time_embed_dim': 256,
        'class_embed_dim': 256,
        'dropout': 0.1,
        'use_content_conditioning': True,
        'content_dim': 512,
        'num_train_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'beta_schedule': 'linear'
    }


if __name__ == '__main__':
    # Test the model
    config = get_default_config()
    model = ECGImg2ImgDiffusion(config)
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 2500)
    timestep = torch.randint(0, 1000, (batch_size,))
    class_label = torch.randint(0, 2, (batch_size,))
    
    noise_pred = model(x, timestep, class_label)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {noise_pred.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
