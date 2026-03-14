"""
Model architectures for Enhanced Diffusion Counterfactual Generation.
Extracted from 16_enhanced_diffusion_cf.py for reusability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ContentEncoder(nn.Module):
    """Extracts class-invariant content from ECG."""
    def __init__(self, in_channels=1, hidden_dim=64, content_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 8, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),
        )
        self.flat_size = hidden_dim * 8 * 8
        self.fc_mu = nn.Linear(self.flat_size, content_dim)
        self.fc_logvar = nn.Linear(self.flat_size, content_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class StyleEncoder(nn.Module):
    """Extracts class-discriminative style from ECG."""
    def __init__(self, in_channels=1, hidden_dim=64, style_dim=128, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.InstanceNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_style = nn.Linear(hidden_dim * 4, style_dim)
        self.classifier = nn.Linear(style_dim, num_classes)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        style = self.fc_style(h)
        class_logits = self.classifier(style)
        return style, class_logits


class ConditionalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhlk,bhck->bhcl', attn, v).reshape(B, C, L)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    """UNet conditioned on timestep, content, style, and target class."""
    def __init__(self, in_ch=1, model_ch=64, content_dim=256, style_dim=128, num_classes=2):
        super().__init__()
        time_dim = model_ch * 4
        cond_dim = time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.content_proj = nn.Linear(content_dim, time_dim)
        self.style_proj = nn.Linear(style_dim, time_dim)
        self.class_embed = nn.Embedding(num_classes + 1, time_dim)
        self.input_conv = nn.Conv1d(in_ch, model_ch, 3, padding=1)
        self.down1 = ConditionalResBlock(model_ch, model_ch, cond_dim)
        self.down2 = ConditionalResBlock(model_ch, model_ch * 2, cond_dim)
        self.down3 = ConditionalResBlock(model_ch * 2, model_ch * 4, cond_dim)
        self.down4 = ConditionalResBlock(model_ch * 4, model_ch * 8, cond_dim)
        self.downsample1 = nn.Conv1d(model_ch, model_ch, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv1d(model_ch * 2, model_ch * 2, 3, stride=2, padding=1)
        self.downsample3 = nn.Conv1d(model_ch * 4, model_ch * 4, 3, stride=2, padding=1)
        self.downsample4 = nn.Conv1d(model_ch * 8, model_ch * 8, 3, stride=2, padding=1)
        self.mid1 = ConditionalResBlock(model_ch * 8, model_ch * 8, cond_dim)
        self.mid_attn = SelfAttention1D(model_ch * 8)
        self.mid2 = ConditionalResBlock(model_ch * 8, model_ch * 8, cond_dim)
        self.up4 = ConditionalResBlock(model_ch * 16, model_ch * 8, cond_dim)
        self.up3 = ConditionalResBlock(model_ch * 8, model_ch * 4, cond_dim)
        self.up2 = ConditionalResBlock(model_ch * 4, model_ch * 2, cond_dim)
        self.up1 = ConditionalResBlock(model_ch * 2, model_ch, cond_dim)
        self.upsample4 = nn.ConvTranspose1d(model_ch * 8, model_ch * 8, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(model_ch * 8, model_ch * 4, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose1d(model_ch * 4, model_ch * 2, 4, stride=2, padding=1)
        self.upsample1 = nn.ConvTranspose1d(model_ch * 2, model_ch, 4, stride=2, padding=1)
        self.out_norm = nn.GroupNorm(32, model_ch)
        self.out_conv = nn.Conv1d(model_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
    def forward(self, x, t, content, style, class_label=None):
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        c_emb = self.content_proj(content)
        s_emb = self.style_proj(style)
        if class_label is not None:
            class_emb = self.class_embed(class_label)
        else:
            class_emb = 0.0
        cond = t_emb + c_emb + s_emb + class_emb
        h = self.input_conv(x)
        h1 = self.down1(h, cond)
        h = self.downsample1(h1)
        h2 = self.down2(h, cond)
        h = self.downsample2(h2)
        h3 = self.down3(h, cond)
        h = self.downsample3(h3)
        h4 = self.down4(h, cond)
        h = self.downsample4(h4)
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        h = self.upsample4(h)
        h = self._match_size(h, h4)
        h = self.up4(torch.cat([h, h4], dim=1), cond)
        h = self.upsample3(h)
        h = self._match_size(h, h3)
        h = self.up3(torch.cat([h, h3], dim=1), cond)
        h = self.upsample2(h)
        h = self._match_size(h, h2)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self._match_size(h, h1)
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        h = F.silu(self.out_norm(h))
        h = self._match_size(h, x)
        return self.out_conv(h)
    
    def _match_size(self, x, target):
        if x.size(-1) != target.size(-1):
            diff = target.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :target.size(-1)]
        return x


class DDIMScheduler:
    """DDIM Scheduler with SDEdit support for partial denoising."""
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, device=device) / num_timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clamp(betas, 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self.sqrt_alphas_cumprod[t][:, None, None] * x_start + 
                self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise)
    
    def sdedit_sample(self, model, content, style, class_label, x_orig, strength=0.6, num_steps=50, cfg_scale=3.0):
        """SDEdit-style sampling: Start from noisy original signal."""
        device = x_orig.device
        start_timestep = int(self.num_timesteps * strength)
        t_start = torch.full((x_orig.size(0),), start_timestep, device=device, dtype=torch.long)
        noise = torch.randn_like(x_orig)
        x = self.q_sample(x_orig, t_start, noise)
        step_size = max(1, start_timestep // num_steps)
        timesteps = list(range(0, start_timestep, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            if cfg_scale > 1.0:
                pred_cond = model(x, t_batch, content, style, class_label)
                uncond_label = torch.full_like(class_label, 2)  # Unconditional class
                pred_uncond = model(x, t_batch, content, style, uncond_label)
                pred_noise = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_noise = model(x, t_batch, content, style, class_label)
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = pred_x0
        return x
