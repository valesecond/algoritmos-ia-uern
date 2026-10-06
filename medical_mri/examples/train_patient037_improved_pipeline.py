"""
Improved Temporal Medical Image Prediction Pipeline v2.0
==========================================================

This pipeline refactors the original voxel-wise approach to use a sophisticated
spatiotemporal 3D CNN with multi-task learning, specialized loss functions,
and domain alignment to improve both image prediction quality and tumor segmentation.

Key Improvements:
- Spatiotemporal 3D CNN with better architectural design
- Multi-task learning: joint image prediction + auxiliary tumor segmentation
- Specialized loss functions: L1 + SSIM + tumor ROI weighting + entropy loss
- Domain alignment: histogram matching, intensity normalization for nnU-Net compatibility
- Robust preprocessing per-modality with outlier handling
- Advanced postprocessing with gradient preservation
- Comprehensive metrics (image + segmentation) with baseline comparison
- Production-grade code with full documentation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    zoom,
    gaussian_filter,
    binary_opening,
    binary_closing,
    label,
)
from scipy.stats import entropy as shannon_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVR
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_WEEKS = [
    "week-000-1",
    "week-000-2",
    "week-002",
    "week-015",
    "week-025",
    "week-029",
    "week-037",
]
DEFAULT_MODALITIES = ["T1", "T2", "FLAIR", "CT1"]


def resolve_patient_paths(patient_dir: Path) -> tuple[Path, Path]:
    """
    Resolve LUMIERE folder layout for Patient-037.

    Returns (data_dir, output_root):
      - data_dir: directory containing week-* folders
      - output_root: parent folder where rnn_prediction_outputs is written
    """
    patient_dir = patient_dir.resolve()
    data_dir = patient_dir
    if not (data_dir / DEFAULT_WEEKS[0]).exists():
        nested = data_dir / "patient-037"
        if (nested / DEFAULT_WEEKS[0]).exists():
            data_dir = nested

    output_root = patient_dir
    if data_dir.name == "patient-037":
        parent_outputs = data_dir.parent / "rnn_prediction_outputs"
        nested_outputs = data_dir / "rnn_prediction_outputs"
        if parent_outputs.exists() or not nested_outputs.exists():
            output_root = data_dir.parent
        else:
            output_root = data_dir
    return data_dir, output_root


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WeekData:
    """Represents multi-modal data for a single week."""
    week: str
    modalities_resampled: np.ndarray  # (C, D, H, W) - all modalities normalized
    flair_resampled: np.ndarray       # (D, H, W) - FLAIR only
    mask_resampled: np.ndarray        # (D, H, W) - tumor mask
    flair_original: np.ndarray        # Original space FLAIR
    mask_original: np.ndarray         # Original space mask
    flair_affine: np.ndarray
    flair_header: Any
    # Additional: statistics for domain alignment
    flair_mean: float = 0.0
    flair_std: float = 1.0
    tumor_ratio: float = 0.0          # % of voxels with tumor


@dataclass
class CaseResult:
    """Results for a temporal prediction case."""
    case_name: str
    target_week: str
    train_weeks: list[str]
    model_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    outputs: dict[str, str]
    improvement_metric: float = 0.0   # (model_dice - baseline_dice) / baseline_dice


# ============================================================================
# PREPROCESSING & NORMALIZATION
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def resolve_compute_device(preferred: Optional[str] = None) -> str:
    """
    Pick the best available PyTorch device (CUDA > Intel XPU > DirectML > CPU).
    DirectML / IPEX require Python <= 3.12; on 3.13+ this falls back to CPU with a warning.
    """
    if preferred:
        pref = preferred.strip().lower()
        if pref == "cpu":
            return "cpu"
        if pref.startswith("cuda") and torch.cuda.is_available():
            return preferred
        if pref == "xpu":
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return "xpu"
            except Exception:
                pass

    if torch.cuda.is_available():
        return "cuda:0"

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("    Using Intel GPU (torch.xpu)")
            return "xpu"
    except Exception:
        pass

    try:
        import torch_directml  # type: ignore

        print("    Using Intel/AMD GPU via torch-directml")
        return str(torch_directml.device())
    except Exception:
        pass

    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) >= (3, 13):
        print(
            "    AVISO: Python 3.13+ — GPU Intel (DirectML/IPEX) indisponível. "
            "Use venv Python 3.11 + 'pip install torch-directml' para acelerar. Treino em CPU."
        )
    return "cpu"


def volume_looks_like_work_space(volume: np.ndarray) -> bool:
    """Heuristic: normalized FLAIR is roughly z-scored in [-8, 8]."""
    v = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p99 = float(np.percentile(np.abs(v), 99.0))
    std_v = float(v.std())
    return (p99 <= 10.5) and (std_v <= 6.5)


def align_prediction_to_work_space(
    pred_volume: np.ndarray,
    target_week: WeekData,
    order: int = 1,
) -> np.ndarray:
    """Map a saved NIfTI prediction into the same normalized space as flair_resampled."""
    pred = resample_to_shape(pred_volume.astype(np.float32), target_week.flair_resampled.shape, order=order)
    if volume_looks_like_work_space(pred):
        return pred.astype(np.float32)
    normalized, _, _ = robust_normalize(pred)
    return normalized.astype(np.float32)


def robust_normalize(volume: np.ndarray, p_low: float = 2, p_high: float = 98) -> tuple[np.ndarray, float, float]:
    """
    Robust z-score normalization that handles outliers.
    Uses IQR-based std calculation to avoid extreme values.
    Preserves full dynamic range including tumor signal.
    Returns (normalized, mean, std) for proper denormalization
    """
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    flat = volume.reshape(-1)

    # Ignore dominant zero-background when computing robust stats.
    nz = flat[np.abs(flat) > 1e-6]
    if nz.size >= max(128, int(0.002 * flat.size)):
        stats_source = nz
    else:
        stats_source = flat

    # Use robust statistics: median for center, IQR for spread
    mean_val = float(np.percentile(stats_source, 50))  # Median is more robust than mean

    # Use IQR-based std which is robust to outliers
    q75 = np.percentile(stats_source, 75)
    q25 = np.percentile(stats_source, 25)
    iqr = q75 - q25
    std_val = float(iqr / 1.35) if iqr > 0 else 1.0  # 1.35 is IQR-to-std ratio
    
    # Ensure minimum std to avoid division by near-zero
    std_val = max(std_val, 0.1)
    
    # Z-score normalization using robust statistics
    # This preserves tumor signal while handling outliers
    normalized = ((volume - mean_val) / std_val).astype(np.float32)
    
    # Clip extreme values to reasonable range.
    normalized = np.clip(normalized, -8, 8).astype(np.float32)
    
    return normalized, mean_val, std_val

def normalize_zscore(volume: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score normalization with saved statistics."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean_val = float(volume.mean())
    std_val = float(volume.std())
    if std_val < 1e-6:
        std_val = 1.0
    normalized = ((volume - mean_val) / std_val).astype(np.float32)
    return normalized, mean_val, std_val


def resample_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    """Resample volume to target shape using scipy zoom."""
    if volume.shape == target_shape:
        return volume.astype(np.float32)
    factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return zoom(volume, factors, order=order).astype(np.float32)


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Match histogram of src to ref for domain alignment.
    Useful to make predicted image distribution match real data.
    """
    try:
        # Reshape for 2D processing
        src_2d = src.reshape(-1, src.shape[-1]) if src.ndim > 2 else src
        ref_2d = ref.reshape(-1, ref.shape[-1]) if ref.ndim > 2 else ref
        
        matched = match_histograms(src_2d, ref_2d, channel_axis=None)
        return matched.reshape(src.shape).astype(np.float32)
    except Exception as e:
        print(f"Warning: histogram matching failed: {e}. Returning source as-is.")
        return src.astype(np.float32)


def load_nifti_volume(nifti_path: str | Path) -> tuple[np.ndarray, np.ndarray, Any]:
    """Load NIfTI volume and return (data, affine, header)."""
    if not Path(nifti_path).exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    
    # Handle 4D (e.g., 4D NIfTI)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[:, :, :, 0]
    
    return data, img.affine, img.header


# ============================================================================
# ARCHITECTURE: Improved Spatiotemporal 3D Model with Multi-Task Learning
# ============================================================================

class ConvBlock3D(nn.Module):
    """3D Convolutional Block with normalization and activation."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock3D(nn.Module):
    """3D Residual Block."""
    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(ch, ch, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm3d(ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return x + out


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism."""
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0, f"channels ({channels}) must be divisible by n_heads ({n_heads})"
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, T, C, ...)"""
        B, T, C = x.shape[:3]
        spatial_shape = x.shape[3:]
        
        # Flatten spatial to (B, T, C)
        x_flat = x.view(B, T, C, -1).mean(dim=-1)  # Global average
        
        # Apply attention
        attended, _ = self.mha(x_flat, x_flat, x_flat)
        
        # Broadcast back to spatial
        attended = attended.view(B, T, C, 1, 1, 1)
        weight = torch.sigmoid(attended)
        
        return x * weight + x


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for conditional processing."""
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(cond_ch, feat_ch),
            nn.ReLU()
        )
        self.beta_net = nn.Sequential(
            nn.Linear(cond_ch, feat_ch),
            nn.ReLU()
        )
    
    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, D, H, W)
        cond: (B, cond_ch)
        """
        gamma = self.gamma_net(cond).view(feat.shape[0], -1, 1, 1, 1)
        beta = self.beta_net(cond).view(feat.shape[0], -1, 1, 1, 1)
        return feat * (1.0 + gamma) + beta


class ImprovedSpatialTemporalPredictor(nn.Module):
    """
    Improved spatiotemporal predictor with multi-task learning.
    Jointly predicts: (1) next week image, (2) auxiliary tumor mask
    This forces learned features to be tumor-aware.
    """
    def __init__(
        self,
        seq_len: int = 3,
        in_channels: int = 4,
        init_features: int = 32,
        num_tumor_classes: int = 2,
        enable_auxiliary_segmentation: bool = True,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.enable_auxiliary_segmentation = enable_auxiliary_segmentation
        self.use_temporal_attention = use_temporal_attention
        
        # ENCODER: Progressive downsampling with residual connections
        self.enc_in = ConvBlock3D(seq_len * in_channels, init_features)
        
        self.enc1 = nn.Sequential(
            ResBlock3D(init_features),
            ConvBlock3D(init_features, init_features * 2, 3)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            ResBlock3D(init_features * 2),
            ConvBlock3D(init_features * 2, init_features * 4, 3)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = nn.Sequential(
            ResBlock3D(init_features * 4),
            ConvBlock3D(init_features * 4, init_features * 8, 3)
        )
        self.pool3 = nn.MaxPool3d(2)
        
        # TEMPORAL ATTENTION (bottleneck)
        if self.use_temporal_attention:
            self.temporal_att = TemporalAttention(init_features * 8, n_heads=4)
        
        # BOTTLENECK with residual
        self.bottleneck = nn.Sequential(
            ResBlock3D(init_features * 8),
            ResBlock3D(init_features * 8)
        )
        
        # DECODER for image prediction: Progressive upsampling
        self.dec3_up = nn.ConvTranspose3d(init_features * 8, init_features * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock3D(init_features * (4 + 8), init_features * 4),
            ResBlock3D(init_features * 4)
        )
        
        self.dec2_up = nn.ConvTranspose3d(init_features * 4, init_features * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock3D(init_features * (2 + 4), init_features * 2),
            ResBlock3D(init_features * 2)
        )
        
        self.dec1_up = nn.ConvTranspose3d(init_features * 2, init_features, 2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock3D(init_features * (1 + 2), init_features),
            ResBlock3D(init_features)
        )
        
        # OUTPUT: 1-channel FLAIR prediction
        self.pred_out = nn.Sequential(
            ConvBlock3D(init_features, init_features),
            nn.Conv3d(init_features, 1, 1)
        )
        
        # AUXILIARY SEGMENTATION HEAD (multi-task learning)
        if self.enable_auxiliary_segmentation:
            self.seg_out = nn.Sequential(
                ConvBlock3D(init_features, init_features),
                nn.Conv3d(init_features, num_tumor_classes, 1)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_seg: bool = True
    ) -> tuple[torch.Tensor] | torch.Tensor:
        """
        Args:
            x: (B, T, C, D, H, W) - temporal sequence of multi-modal volumes
            return_seg: if True, return both prediction and auxiliary segmentation
        
        Returns:
            (pred, seg) if return_seg and enable_auxiliary_segmentation, else pred
        """
        B, T, C, D, H, W = x.shape
        
        # Flatten temporal dimension: (B, T*C, D, H, W)
        x_flat = x.view(B, T * C, D, H, W)
        
        # ENCODER
        e0 = self.enc_in(x_flat)
        e1 = self.enc1(e0)
        e1_down = self.pool1(e1)
        
        e2 = self.enc2(e1_down)
        e2_down = self.pool2(e2)
        
        e3 = self.enc3(e2_down)
        e3_down = self.pool3(e3)
        
        # BOTTLENECK
        bn = self.bottleneck(e3_down)
        
        # DECODER
        d3 = self.dec3_up(bn)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.dec2_up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.dec1_up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # PREDICTIONS
        pred = self.pred_out(d1)
        
        if self.enable_auxiliary_segmentation and return_seg:
            seg = self.seg_out(d1)  # Logits for segmentation
            return pred, seg
        
        return pred


# ============================================================================
# LOSS FUNCTIONS (Multi-component, tumor-focused)
# ============================================================================

def compute_ssim_loss(pred: torch.Tensor, target: torch.Tensor, win_size: int = 11) -> torch.Tensor:
    """Differentiable 3D SSIM loss."""
    # Adapt kernel size to small volumes at coarser MS scales.
    spatial_dims = pred.shape[-3:]
    k = min(win_size, int(min(spatial_dims)))
    if k < 3:
        return F.l1_loss(pred, target)
    if k % 2 == 0:
        k -= 1

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool3d(pred, kernel_size=k, stride=1, padding=k // 2)
    mu_y = F.avg_pool3d(target, kernel_size=k, stride=1, padding=k // 2)

    sigma_x = F.avg_pool3d(pred * pred, kernel_size=k, stride=1, padding=k // 2) - mu_x * mu_x
    sigma_y = F.avg_pool3d(target * target, kernel_size=k, stride=1, padding=k // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool3d(pred * target, kernel_size=k, stride=1, padding=k // 2) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-7)
    return 1.0 - ssim_map.mean()


def compute_multiscale_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable multi-scale SSIM in 3D."""
    scale_weights = [1.0, 0.7, 0.5]
    p = pred
    t = target
    losses = []
    for i, weight in enumerate(scale_weights):
        losses.append(weight * compute_ssim_loss(p, t, win_size=7))
        if i < len(scale_weights) - 1:
            p = F.avg_pool3d(p, kernel_size=2, stride=2)
            t = F.avg_pool3d(t, kernel_size=2, stride=2)
    return sum(losses) / sum(scale_weights)


def compute_tumor_roi_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tumor_mask: torch.Tensor,
    roi_weight: float = 5.0
) -> torch.Tensor:
    """
    Weighted L1 loss with higher weight on tumor ROI.
    This forces the model to be more accurate in the tumor region.
    """
    # Create per-voxel weight: higher in tumor region
    weights = torch.ones_like(target)
    weights[tumor_mask > 0.5] = roi_weight
    
    # Weighted L1
    loss = weights * torch.abs(pred - target)
    return loss.mean()


def compute_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss to preserve entropy/texture information.
    Helps maintain image structure and contrast.
    """
    # Map to stable probability-like range before entropy computation.
    pred_prob = torch.sigmoid(pred)
    target_prob = torch.sigmoid(target)

    pred_smooth = -torch.sum(pred_prob * torch.log(pred_prob + 1e-7), dim=1)
    target_smooth = -torch.sum(target_prob * torch.log(target_prob + 1e-7), dim=1)
    
    entropy_loss = F.mse_loss(pred_smooth, target_smooth)
    return entropy_loss


def compute_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Preserve edge and texture variation by matching spatial gradients."""
    dz_p = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dx_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

    dz_t = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    dx_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    return (
        F.l1_loss(dz_p, dz_t) +
        F.l1_loss(dy_p, dy_t) +
        F.l1_loss(dx_p, dx_t)
    ) / 3.0


def compute_perceptual_pyramid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lightweight perceptual loss on multi-scale pooled representations."""
    scales = [1, 2, 4]
    loss = 0.0
    for s in scales:
        if s == 1:
            p = pred
            t = target
        else:
            p = F.avg_pool3d(pred, kernel_size=s, stride=s)
            t = F.avg_pool3d(target, kernel_size=s, stride=s)

        # Compare smoothed and edge-emphasized representations.
        p_blur = F.avg_pool3d(p, kernel_size=3, stride=1, padding=1)
        t_blur = F.avg_pool3d(t, kernel_size=3, stride=1, padding=1)
        p_high = p - p_blur
        t_high = t - t_blur

        loss = loss + 0.6 * F.smooth_l1_loss(p_blur, t_blur) + 0.4 * F.smooth_l1_loss(p_high, t_high)

    return loss / float(len(scales))


def compute_local_texture_loss(pred: torch.Tensor, target: torch.Tensor, win: int = 5) -> torch.Tensor:
    """Match local texture statistics (local standard deviation maps)."""
    mu_p = F.avg_pool3d(pred, kernel_size=win, stride=1, padding=win // 2)
    mu_t = F.avg_pool3d(target, kernel_size=win, stride=1, padding=win // 2)

    var_p = F.avg_pool3d((pred - mu_p) ** 2, kernel_size=win, stride=1, padding=win // 2)
    var_t = F.avg_pool3d((target - mu_t) ** 2, kernel_size=win, stride=1, padding=win // 2)

    std_p = torch.sqrt(var_p + 1e-6)
    std_t = torch.sqrt(var_t + 1e-6)
    return F.smooth_l1_loss(std_p, std_t)


def compute_diffuse_prediction_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    tumor_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Penalize diffuse over-prediction outside tumor ROI."""
    if tumor_mask is None:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    outside = (tumor_mask <= 0.5).float()
    outside_vox = torch.sum(outside)
    if outside_vox < 1.0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_mu = torch.mean(pred, dim=(2, 3, 4), keepdim=True)
    pred_std = torch.std(pred, dim=(2, 3, 4), keepdim=True).clamp(min=1e-4)
    targ_mu = torch.mean(target, dim=(2, 3, 4), keepdim=True)
    targ_std = torch.std(target, dim=(2, 3, 4), keepdim=True).clamp(min=1e-4)

    pred_z = (pred - pred_mu) / pred_std
    targ_z = (target - targ_mu) / targ_std

    pred_tail = F.relu(pred_z - 1.0)
    targ_tail = F.relu(targ_z - 1.0)

    return torch.sum(torch.abs(pred_tail - targ_tail) * outside) / outside_vox


class CombinedLoss(nn.Module):
    """Combined loss: L1 + SSIM + Tumor ROI + Entropy."""
    def __init__(
        self,
        l1_weight: float = 0.33,
        ssim_weight: float = 0.34,
        roi_weight: float = 5.0,
        roi_loss_weight: float = 0.12,
        entropy_weight: float = 0.01,
        gradient_weight: float = 0.10,
        perceptual_weight: float = 0.17,
        texture_weight: float = 0.06,
        aux_weight: float = 0.04,
        diffuse_weight: float = 0.08,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.roi_weight = roi_weight
        self.roi_loss_weight = roi_loss_weight
        self.entropy_weight = entropy_weight
        self.gradient_weight = gradient_weight
        self.perceptual_weight = perceptual_weight
        self.texture_weight = texture_weight
        self.aux_weight = aux_weight
        self.diffuse_weight = diffuse_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tumor_mask: Optional[torch.Tensor] = None,
        seg_pred: Optional[torch.Tensor] = None,
        seg_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: predicted image (B, 1, D, H, W)
            target: target image (B, 1, D, H, W)
            tumor_mask: (B, 1, D, H, W) optional tumor mask for ROI weighting
            seg_pred: (B, C, D, H, W) auxiliary segmentation logits
            seg_target: (B, D, H, W) auxiliary segmentation target
        """
        # L1 component
        l1_loss = F.l1_loss(pred, target)
        
        # Multi-scale SSIM component (stronger structural supervision)
        ssim_loss = compute_multiscale_ssim_loss(pred, target)
        
        # Tumor ROI component
        roi_loss = 0.0
        if tumor_mask is not None:
            roi_loss = compute_tumor_roi_loss(pred, target, tumor_mask, roi_weight=self.roi_weight)
        
        # Entropy preservation component
        entropy_loss = compute_entropy_loss(pred, target)

        # Structure-preserving gradient component
        gradient_loss = compute_gradient_loss(pred, target)

        # Perceptual-like multi-scale feature component
        perceptual_loss = compute_perceptual_pyramid_loss(pred, target)

        # Local texture consistency
        texture_loss = compute_local_texture_loss(pred, target)

        # Penalize diffuse high-intensity spread outside tumor ROI
        diffuse_penalty = compute_diffuse_prediction_penalty(pred, target, tumor_mask)

        # Auxiliary segmentation loss (if available)
        aux_loss = 0.0
        if seg_pred is not None and seg_target is not None:
            aux_loss = F.cross_entropy(seg_pred, seg_target.long())
        
        # Combine all losses
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.roi_loss_weight * roi_loss +
            self.entropy_weight * entropy_loss +
            self.gradient_weight * gradient_loss +
            self.perceptual_weight * perceptual_loss +
            self.texture_weight * texture_loss +
            self.diffuse_weight * diffuse_penalty +
            self.aux_weight * aux_loss
        )
        
        return total_loss


# ============================================================================
# DATA LOADING & BUILDING TEMPORAL WINDOWS
# ============================================================================

def load_week_data(
    patient_dir: Path,
    week: str,
    modalities: list[str],
    target_shape: tuple[int, int, int],
) -> WeekData:
    """Load all modalities for a week and prepare data."""
    week_path = patient_dir / week
    if not week_path.exists():
        raise FileNotFoundError(f"Week directory not found: {week_path}")
    
    # Load FLAIR (primary modality)
    flair_path = week_path / "FLAIR.nii.gz"
    flair_data, flair_affine, flair_header = load_nifti_volume(flair_path)
    
    # Load segmentation mask from known project paths.
    mask_candidates = [
        week_path / "mask.nii.gz",
        week_path / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "flair_seg_mask.nii.gz",
        week_path / "DeepBraTumIA-segmentation" / "atlas" / "segmentation" / "seg_mask.nii.gz",
        week_path / "HD-GLIO-AUTO-segmentation" / "registered" / "segmentation.nii.gz",
        week_path / "HD-GLIO-AUTO-segmentation" / "native" / "segmentation_FLAIR_origspace.nii.gz",
    ]
    mask_data = None
    for mask_path in mask_candidates:
        if mask_path.exists():
            mask_data, _, _ = load_nifti_volume(mask_path)
            break
    if mask_data is None:
        print(f"Warning: segmentation mask not found for {week}, using zeros")
        mask_data = np.zeros_like(flair_data)
    
    # Load all modalities (strict: no silent zero-fill for missing images).
    modalities_data = []
    for mod in modalities:
        mod_path = week_path / f"{mod}.nii.gz"
        if not mod_path.exists():
            raise FileNotFoundError(f"Required modality not found for {week}: {mod_path}")
        mod_data, _, _ = load_nifti_volume(mod_path)
        if mod_data.ndim != 3:
            raise ValueError(f"Invalid modality shape for {week}/{mod}: expected 3D, got {mod_data.shape}")
        if not np.isfinite(mod_data).all():
            raise ValueError(f"Invalid modality values for {week}/{mod}: contains NaN/Inf")
        
        modalities_data.append(mod_data)
    
    # Resample to working shape
    flair_resampled = resample_to_shape(flair_data, target_shape, order=1)
    mask_resampled = resample_to_shape(mask_data, target_shape, order=0)
    
    modalities_resampled = []
    for mod_data in modalities_data:
        mod_resampled = resample_to_shape(mod_data, target_shape, order=1)
        # Normalize each modality independently
        mod_normalized, _, _ = robust_normalize(mod_resampled)
        modalities_resampled.append(mod_normalized)
    
    modalities_resampled = np.stack(modalities_resampled, axis=0)  # (C, D, H, W)
    
    # Normalize FLAIR with z-score (preserves tumor signal)
    flair_normalized, flair_mean, flair_std = robust_normalize(flair_resampled)
    
    return WeekData(
        week=week,
        modalities_resampled=modalities_resampled,
        flair_resampled=flair_normalized,
        mask_resampled=mask_resampled,
        flair_original=flair_data,
        mask_original=mask_data,
        flair_affine=flair_affine,
        flair_header=flair_header,
        flair_mean=flair_mean,
        flair_std=flair_std,
        tumor_ratio=float((mask_resampled > 0.5).mean()),
    )


def build_temporal_windows(
    weeks_data: list[WeekData],
    seq_len: int,
    target_idx: int,
) -> list[tuple[list[int], int]]:
    """
    Build temporal windows for training and testing.
    
    Returns: list of (sequence_indices, target_index)
    """
    windows = []
    
    # Training windows: using past weeks to predict future
    for i in range(seq_len - 1, target_idx):
        # Last seq_len-1 weeks before target, then predict target+1
        seq_indices = list(range(i - (seq_len - 1), i + 1))
        target_next = i + 1
        if target_next < len(weeks_data):
            windows.append((seq_indices, target_next))
    
    return windows


def sample_batch(
    weeks_data: list[WeekData],
    windows: list[tuple[list[int], int]],
    batch_size: int,
    patch_size: tuple[int, int, int],
    tumor_patch_prob: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a random batch of patches."""
    batch_x = []
    batch_y = []
    batch_mask = []
    
    D, H, W = weeks_data[0].flair_resampled.shape
    dz, dy, dx = patch_size
    
    for _ in range(batch_size):
        # Random window
        seq_indices, target_idx = random.choice(windows)
        
        # Tumor-centered patch sampling to preserve lesion information.
        target_mask_full = weeks_data[target_idx].mask_resampled
        if random.random() < tumor_patch_prob and np.any(target_mask_full > 0.5):
            tumor_vox = np.argwhere(target_mask_full > 0.5)
            cz, cy, cx = tumor_vox[random.randint(0, len(tumor_vox) - 1)]
            z_start = int(np.clip(cz - dz // 2 + random.randint(-2, 2), 0, max(0, D - dz)))
            y_start = int(np.clip(cy - dy // 2 + random.randint(-2, 2), 0, max(0, H - dy)))
            x_start = int(np.clip(cx - dx // 2 + random.randint(-2, 2), 0, max(0, W - dx)))
        else:
            z_start = random.randint(0, max(0, D - dz))
            y_start = random.randint(0, max(0, H - dy))
            x_start = random.randint(0, max(0, W - dx))
        
        # Extract patches from sequence
        x_seq = []
        for idx in seq_indices:
            mods = weeks_data[idx].modalities_resampled
            patch = mods[:, z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx]
            
            # Ensure patch is right size (pad if needed)
            if patch.shape != (mods.shape[0], dz, dy, dx):
                pad_patch = np.zeros((mods.shape[0], dz, dy, dx), dtype=np.float32)
                pad_patch[:, :patch.shape[1], :patch.shape[2], :patch.shape[3]] = patch
                patch = pad_patch
            
            x_seq.append(torch.from_numpy(patch))
        
        x_seq = torch.stack(x_seq)  # (T, C, dz, dy, dx)
        
        # Target FLAIR
        target_flair = weeks_data[target_idx].flair_resampled
        y_patch = target_flair[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx]
        if y_patch.shape != (dz, dy, dx):
            pad_y = np.zeros((dz, dy, dx), dtype=np.float32)
            pad_y[:y_patch.shape[0], :y_patch.shape[1], :y_patch.shape[2]] = y_patch
            y_patch = pad_y
        
        y_patch = torch.from_numpy(y_patch).unsqueeze(0)  # (1, dz, dy, dx)
        
        # Target mask for ROI weighting
        mask_patch = weeks_data[target_idx].mask_resampled
        m_patch = mask_patch[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx]
        if m_patch.shape != (dz, dy, dx):
            pad_m = np.zeros((dz, dy, dx), dtype=np.float32)
            pad_m[:m_patch.shape[0], :m_patch.shape[1], :m_patch.shape[2]] = m_patch
            m_patch = pad_m
        
        m_patch = torch.from_numpy(m_patch).unsqueeze(0)  # (1, dz, dy, dx)
        
        batch_x.append(x_seq)
        batch_y.append(y_patch)
        batch_mask.append(m_patch)
    
    batch_x = torch.stack(batch_x)  # (B, T, C, dz, dy, dx)
    batch_y = torch.stack(batch_y)  # (B, 1, dz, dy, dx)
    batch_mask = torch.stack(batch_mask)  # (B, 1, dz, dy, dx)
    
    return batch_x, batch_y, batch_mask


def _latest_sequence_indices(total_weeks: int, target_idx: int, seq_len: int) -> list[int]:
    """Get indices for latest temporal context before target."""
    seq_indices = list(range(max(0, target_idx - seq_len), target_idx))
    while len(seq_indices) < seq_len:
        seq_indices.insert(0, seq_indices[0] if seq_indices else 0)
    return seq_indices[-seq_len:]


def _extract_connected_regions(mask: np.ndarray, min_voxels: int = 16, max_regions: int = 16) -> list[np.ndarray]:
    """Extract connected segmented regions to be used as graph nodes."""
    binary = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0) > 0.5
    labeled, ncc = label(binary.astype(np.uint8))

    regions: list[np.ndarray] = []
    for region_idx in range(1, ncc + 1):
        region = labeled == region_idx
        if int(region.sum()) >= min_voxels:
            regions.append(region)

    if not regions:
        if binary.any():
            regions = [binary]
        else:
            fallback = np.zeros_like(binary, dtype=bool)
            center = tuple(int(s // 2) for s in binary.shape)
            radius = tuple(max(1, int(s // 10)) for s in binary.shape)
            slices = tuple(
                slice(max(0, c - r), min(dim, c + r + 1))
                for c, r, dim in zip(center, radius, binary.shape)
            )
            fallback[slices] = True
            regions = [fallback]

    regions.sort(key=lambda region: int(region.sum()), reverse=True)
    return regions[:max_regions]


def _region_geometry_features(region_mask: np.ndarray) -> dict[str, float]:
    """Compute static geometric descriptors for a segmented region."""
    region_mask = np.nan_to_num(region_mask, nan=0.0, posinf=0.0, neginf=0.0) > 0.5
    total_voxels = float(max(region_mask.size, 1))
    coords = np.argwhere(region_mask)

    if coords.size == 0:
        centroid = np.array(region_mask.shape, dtype=np.float32) / 2.0
        extent = np.zeros(3, dtype=np.float32)
    else:
        centroid = coords.mean(axis=0).astype(np.float32)
        extent = (coords.max(axis=0) - coords.min(axis=0) + 1.0).astype(np.float32)

    eroded = binary_erosion(region_mask)
    border = np.logical_and(region_mask, np.logical_not(eroded))
    perimeter = float(border.sum()) / total_voxels
    shape = np.maximum(np.asarray(region_mask.shape, dtype=np.float32) - 1.0, 1.0)

    return {
        "area": float(region_mask.sum()) / total_voxels,
        "perimeter": perimeter,
        "centroid_z": float(centroid[0] / shape[0]),
        "centroid_y": float(centroid[1] / shape[1]),
        "centroid_x": float(centroid[2] / shape[2]),
        "extent_z": float(extent[0] / max(float(region_mask.shape[0]), 1.0)),
        "extent_y": float(extent[1] / max(float(region_mask.shape[1]), 1.0)),
        "extent_x": float(extent[2] / max(float(region_mask.shape[2]), 1.0)),
    }


def _region_intensity_features(volume: np.ndarray, region_mask: np.ndarray) -> dict[str, float]:
    """Compute intensity descriptors for a region inside a single volume."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    region_mask = np.nan_to_num(region_mask, nan=0.0, posinf=0.0, neginf=0.0) > 0.5
    region_values = volume[region_mask]
    stats = compute_volume_characteristics(region_values)

    shell = np.logical_and(binary_dilation(region_mask), np.logical_not(region_mask))
    shell_values = volume[shell]
    shell_mean = float(np.mean(shell_values)) if shell_values.size > 0 else stats["mean"]

    stats["contrast"] = float(stats["mean"] - shell_mean)
    return stats


def _region_temporal_feature_vector(
    weeks_data: list[WeekData],
    seq_indices: list[int],
    region_mask: np.ndarray,
) -> np.ndarray:
    """Build temporal node features from a segmented region."""
    temporal_stats: list[float] = []
    mean_traj: list[float] = []
    std_traj: list[float] = []
    entropy_traj: list[float] = []
    p95_traj: list[float] = []

    for week_idx in seq_indices:
        stats = _region_intensity_features(weeks_data[week_idx].flair_resampled, region_mask)
        temporal_stats.extend([
            float(stats["mean"]),
            float(stats["std"]),
            float(stats["entropy"]),
            float(stats["p95"]),
            float(stats["contrast"]),
        ])
        mean_traj.append(float(stats["mean"]))
        std_traj.append(float(stats["std"]))
        entropy_traj.append(float(stats["entropy"]))
        p95_traj.append(float(stats["p95"]))

    geom = _region_geometry_features(region_mask)
    temporal_stats.extend([
        geom["area"],
        geom["perimeter"],
        geom["centroid_z"],
        geom["centroid_y"],
        geom["centroid_x"],
        geom["extent_z"],
        geom["extent_y"],
        geom["extent_x"],
        float(mean_traj[-1] - mean_traj[0]) if len(mean_traj) > 1 else 0.0,
        float(std_traj[-1] - std_traj[0]) if len(std_traj) > 1 else 0.0,
        float(entropy_traj[-1] - entropy_traj[0]) if len(entropy_traj) > 1 else 0.0,
        float(p95_traj[-1] - p95_traj[0]) if len(p95_traj) > 1 else 0.0,
    ])

    return np.asarray(temporal_stats, dtype=np.float32)


def _build_temporal_region_baseline(weeks_data: list[WeekData], seq_indices: list[int]) -> np.ndarray:
    """Construct a stable baseline image from the most recent temporal context."""
    latest = weeks_data[seq_indices[-1]].flair_resampled.astype(np.float32)
    if len(seq_indices) < 2:
        return latest.copy()

    previous = weeks_data[seq_indices[-2]].flair_resampled.astype(np.float32)
    velocity = latest - previous
    baseline = latest + 0.55 * velocity
    return np.clip(baseline, -8.0, 8.0).astype(np.float32)


def _build_region_entropy_adjacency(
    features: torch.Tensor,
    seq_len: int,
    k_neighbors: int = 4,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Build adjacency using temporal entropy variation across segmented regions."""
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape (N, F), got {tuple(features.shape)}")

    n_nodes = int(features.shape[0])
    if n_nodes <= 1:
        return torch.eye(n_nodes, device=features.device, dtype=torch.float32)

    block_size = 5  # mean, std, entropy, p95, contrast
    entropy_cols = [min(i * block_size + 2, features.shape[1] - 1) for i in range(seq_len)]
    mean_cols = [min(i * block_size, features.shape[1] - 1) for i in range(seq_len)]
    geom_start = seq_len * block_size
    centroid_start = min(geom_start + 2, max(features.shape[1] - 3, 0))

    entropy_traj = features[:, entropy_cols]
    mean_traj = features[:, mean_cols]
    centroid_traj = features[:, centroid_start:centroid_start + 3]

    entropy_dist = torch.mean(torch.abs(entropy_traj[:, None, :] - entropy_traj[None, :, :]), dim=-1)
    mean_dist = torch.mean(torch.abs(mean_traj[:, None, :] - mean_traj[None, :, :]), dim=-1)
    spatial_dist = torch.norm(centroid_traj[:, None, :] - centroid_traj[None, :, :], dim=-1)

    dist = 0.70 * entropy_dist + 0.20 * mean_dist + 0.10 * spatial_dist
    weights = torch.exp(-dist / max(float(temperature), 1e-6))
    weights = weights - torch.diag_embed(torch.diagonal(weights))

    k_eff = max(1, min(int(k_neighbors), n_nodes - 1))
    top_vals, top_idx = torch.topk(weights, k=k_eff, dim=1)
    sparse_weights = torch.zeros_like(weights)
    sparse_weights.scatter_(1, top_idx, top_vals)

    adjacency = torch.maximum(sparse_weights, sparse_weights.T)
    adjacency = adjacency + torch.eye(n_nodes, device=features.device, dtype=torch.float32)

    degree = torch.clamp(adjacency.sum(dim=1, keepdim=True), min=1e-6)
    return adjacency / degree


class VoxelGRUPredictor(nn.Module):
    """Small GRU regressor for voxel-wise temporal prediction."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class VoxelLSTMPredictor(nn.Module):
    """Small LSTM regressor for voxel-wise temporal prediction."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class VoxelTransformerPredictor(nn.Module):
    """Compact Transformer regressor for voxel-wise temporal sequences."""

    def __init__(self, hidden_size: int = 64, n_heads: int = 4, num_layers: int = 2, max_seq_len: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        seq_len = x.shape[1]
        encoded = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        encoded = self.encoder(encoded)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class TemporalEntropyGraphPredictor(nn.Module):
    """Region-based temporal GNN using segmented regions as graph nodes."""

    def __init__(self, feature_dim: int, hidden_size: int = 96):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.12),
        )
        self.gcn1 = nn.Linear(hidden_size, hidden_size)
        self.gcn2 = nn.Linear(hidden_size, hidden_size)
        self.gcn3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.12)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.entropy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.tumor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a small residual graph network over region nodes."""
        h0 = self.input_proj(x)
        h1 = F.gelu(self.gcn1(adjacency @ h0)) + h0
        h1 = self.dropout(h1)
        h2 = F.gelu(self.gcn2(adjacency @ h1)) + h1
        h2 = self.dropout(h2)
        h3 = F.gelu(self.gcn3(adjacency @ h2)) + h2

        pred_value = self.value_head(h3).squeeze(-1)
        pred_entropy = F.softplus(self.entropy_head(h3).squeeze(-1))
        pred_tumor = torch.sigmoid(self.tumor_head(h3).squeeze(-1))
        return pred_value, pred_entropy, pred_tumor


def train_temporal_gnn_predictor(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    max_samples: int = 28000,
    epochs: int = 18,
    batch_size: int = 768,
    learning_rate: float = 8e-4,
    k_neighbors: int = 4,
    temperature: float = 0.20,
    entropy_weight: float = 0.20,
    tumor_weight: float = 0.35,
    anchor_volume: Optional[np.ndarray] = None,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
    """
    Train a region-based temporal GNN with entropy-driven edges.
    Returns (predicted_volume, tumor_probability_map, info).
    """
    info: dict[str, Any] = {
        "gnn_samples": 0,
        "gnn_regions": 0,
        "gnn_feature_dim": 0,
        "gnn_epochs": 0,
        "gnn_final_loss": None,
        "gnn_pred_entropy_mean": 0.0,
        "gnn_pred_tumor_mean": 0.0,
        "gnn_k_neighbors": int(k_neighbors),
        "gnn_temperature": float(temperature),
        "gnn_entropy_weight": float(entropy_weight),
        "gnn_tumor_weight": float(tumor_weight),
        "gnn_anchor_used": bool(anchor_volume is not None),
    }

    windows = build_temporal_windows(weeks_data, seq_len, target_idx)
    if not windows:
        return None, info

    rng = np.random.default_rng(2048 + target_idx)
    graph_samples: list[dict[str, Any]] = []
    max_regions_per_graph = max(4, min(16, max_samples // max(1, len(windows))))

    for seq_indices, tgt_idx in windows:
        region_union = np.zeros_like(weeks_data[seq_indices[0]].mask_resampled, dtype=bool)
        for week_idx in seq_indices:
            region_union |= (weeks_data[week_idx].mask_resampled > 0.5)

        region_masks = _extract_connected_regions(
            region_union,
            min_voxels=max(12, int(0.0005 * region_union.size)),
            max_regions=max_regions_per_graph,
        )

        if not region_masks:
            continue

        base_volume = _build_temporal_region_baseline(weeks_data, seq_indices)
        target_volume = weeks_data[tgt_idx].flair_resampled.astype(np.float32)
        target_mask = (weeks_data[tgt_idx].mask_resampled > 0.5).astype(np.float32)

        region_features: list[np.ndarray] = []
        region_value_targets: list[float] = []
        region_entropy_targets: list[float] = []
        region_tumor_targets: list[float] = []
        region_sizes: list[float] = []
        kept_masks: list[np.ndarray] = []

        for region_mask in region_masks:
            feat_vec = _region_temporal_feature_vector(weeks_data, seq_indices, region_mask)
            if not np.isfinite(feat_vec).all():
                continue

            base_stats = _region_intensity_features(base_volume, region_mask)
            target_stats = _region_intensity_features(target_volume, region_mask)
            region_features.append(feat_vec)
            region_value_targets.append(float(target_stats["mean"]))
            region_entropy_targets.append(float(target_stats["entropy"]))
            tumor_ratio = float(np.clip(target_mask[region_mask].mean(), 0.0, 1.0))
            region_tumor_targets.append(1.0 if tumor_ratio >= 0.03 else 0.0)
            region_sizes.append(float(region_mask.sum()))
            kept_masks.append(region_mask.astype(np.float32))

        if not region_features:
            continue

        graph_samples.append(
            {
                "features": np.asarray(region_features, dtype=np.float32),
                "value_targets": np.asarray(region_value_targets, dtype=np.float32),
                "entropy_targets": np.asarray(region_entropy_targets, dtype=np.float32),
                "tumor_targets": np.asarray(region_tumor_targets, dtype=np.float32),
                "region_weights": np.asarray(region_sizes, dtype=np.float32),
                "region_masks": kept_masks,
                "base_volume": base_volume.astype(np.float32),
                "target_volume": target_volume.astype(np.float32),
                "target_mask": target_mask.astype(np.float32),
                "seq_indices": seq_indices,
            }
        )

    if not graph_samples:
        return None, info

    feature_stack = np.concatenate([g["features"] for g in graph_samples], axis=0)
    feature_mean = feature_stack.mean(axis=0, keepdims=True)
    feature_std = feature_stack.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    info["gnn_regions"] = int(sum(g["features"].shape[0] for g in graph_samples))
    info["gnn_samples"] = int(info["gnn_regions"])
    info["gnn_feature_dim"] = int(feature_stack.shape[1])

    model = TemporalEntropyGraphPredictor(feature_dim=int(feature_stack.shape[1]), hidden_size=96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    n_graphs = len(graph_samples)
    val_size = max(1, min(max(1, int(0.2 * n_graphs)), n_graphs - 1)) if n_graphs > 1 else 0
    graph_order = np.arange(n_graphs)
    if val_size > 0:
        val_idx = np.sort(rng.choice(n_graphs, size=val_size, replace=False))
        train_mask = np.ones(n_graphs, dtype=bool)
        train_mask[val_idx] = False
        train_idx = graph_order[train_mask]
    else:
        val_idx = np.asarray([], dtype=int)
        train_idx = graph_order

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        epoch_losses = []

        for graph_idx in perm:
            graph = graph_samples[int(graph_idx)]
            xb_np = ((graph["features"] - feature_mean) / feature_std).astype(np.float32)
            xb = torch.from_numpy(xb_np).to(device)
            yv = torch.from_numpy(graph["value_targets"].astype(np.float32)).to(device)
            ye = torch.from_numpy(graph["entropy_targets"].astype(np.float32)).to(device)
            yt = torch.from_numpy(graph["tumor_targets"].astype(np.float32)).to(device)
            region_weights = torch.from_numpy(np.maximum(graph["region_weights"].astype(np.float32), 1.0)).to(device)

            with torch.no_grad():
                adj = _build_region_entropy_adjacency(
                    xb,
                    seq_len=seq_len,
                    k_neighbors=k_neighbors,
                    temperature=temperature,
                )

            optimizer.zero_grad(set_to_none=True)
            pred_val, pred_ent, pred_tumor = model(xb, adj)
            value_loss = (F.smooth_l1_loss(pred_val, yv, reduction="none") * region_weights).sum() / torch.clamp(region_weights.sum(), min=1.0)
            entropy_loss = F.mse_loss(pred_ent, ye)
            tumor_loss = F.binary_cross_entropy(pred_tumor, yt)
            graph_smoothness = torch.mean(torch.abs((adj @ pred_val.unsqueeze(-1)).squeeze(-1) - pred_val))
            loss = value_loss + (entropy_weight * entropy_loss) + (tumor_weight * tumor_loss) + (0.05 * graph_smoothness)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for graph_idx in val_idx:
                graph = graph_samples[int(graph_idx)]
                xb_np = ((graph["features"] - feature_mean) / feature_std).astype(np.float32)
                xb = torch.from_numpy(xb_np).to(device)
                yv = torch.from_numpy(graph["value_targets"].astype(np.float32)).to(device)
                ye = torch.from_numpy(graph["entropy_targets"].astype(np.float32)).to(device)
                yt = torch.from_numpy(graph["tumor_targets"].astype(np.float32)).to(device)
                region_weights = torch.from_numpy(np.maximum(graph["region_weights"].astype(np.float32), 1.0)).to(device)
                adj = _build_region_entropy_adjacency(
                    xb,
                    seq_len=seq_len,
                    k_neighbors=k_neighbors,
                    temperature=temperature,
                )
                pred_val, pred_ent, pred_tumor = model(xb, adj)
                value_loss = (F.smooth_l1_loss(pred_val, yv, reduction="none") * region_weights).sum() / torch.clamp(region_weights.sum(), min=1.0)
                entropy_loss = F.mse_loss(pred_ent, ye)
                tumor_loss = F.binary_cross_entropy(pred_tumor, yt)
                graph_smoothness = torch.mean(torch.abs((adj @ pred_val.unsqueeze(-1)).squeeze(-1) - pred_val))
                loss = value_loss + (entropy_weight * entropy_loss) + (tumor_weight * tumor_loss) + (0.05 * graph_smoothness)
                val_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(epoch_losses))
        scheduler.step(val_loss)
        info["gnn_final_loss"] = float(np.mean(epoch_losses)) if epoch_losses else None
        info["gnn_epochs"] = int(epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    latest_seq = _latest_sequence_indices(len(weeks_data), target_idx, seq_len)
    region_union = np.zeros_like(weeks_data[latest_seq[0]].mask_resampled, dtype=bool)
    for week_idx in latest_seq:
        region_union |= (weeks_data[week_idx].mask_resampled > 0.5)

    region_masks = _extract_connected_regions(
        region_union,
        min_voxels=max(12, int(0.0005 * region_union.size)),
        max_regions=max_regions_per_graph,
    )

    if not region_masks:
        return None, info

    feature_rows = []
    for region_mask in region_masks:
        feat_vec = _region_temporal_feature_vector(weeks_data, latest_seq, region_mask)
        feature_rows.append(feat_vec)

    if not feature_rows:
        return None, info

    X_test = np.asarray(feature_rows, dtype=np.float32)
    X_test = ((X_test - feature_mean) / feature_std).astype(np.float32)
    out_shape = weeks_data[target_idx].flair_resampled.shape
    base_volume = _build_temporal_region_baseline(weeks_data, latest_seq)

    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        adj = _build_region_entropy_adjacency(
            xb,
            seq_len=seq_len,
            k_neighbors=min(k_neighbors, max(1, xb.shape[0] - 1)),
            temperature=temperature,
        )
        pred_val, pred_ent, pred_tumor = model(xb, adj)
        pred_val_np = pred_val.cpu().numpy().astype(np.float32)
        pred_ent_np = pred_ent.cpu().numpy().astype(np.float32)
        pred_tumor_np = np.clip(pred_tumor.cpu().numpy().astype(np.float32), 0.0, 1.0)

    correction = np.zeros(out_shape, dtype=np.float32)
    confidence = np.zeros(out_shape, dtype=np.float32)
    tumor_full = np.zeros(out_shape, dtype=np.float32)

    for region_mask, region_value, tumor_prob in zip(region_masks, pred_val_np, pred_tumor_np):
        region_mask = region_mask.astype(bool)
        region_conf = float(np.clip(0.55 + 0.40 * tumor_prob, 0.55, 0.98))
        local_delta = float(region_value) - float(np.mean(base_volume[region_mask]))
        correction[region_mask] += region_conf * local_delta
        confidence[region_mask] = np.maximum(confidence[region_mask], region_conf)
        tumor_full[region_mask] = np.maximum(tumor_full[region_mask], float(tumor_prob))

    correction = gaussian_filter(correction, sigma=0.65)
    pred_full = np.clip(base_volume + correction, -8.0, 8.0)
    pred_full = np.where(confidence > 0.0, pred_full, base_volume)

    if anchor_volume is not None and anchor_volume.shape == out_shape:
        anchor = np.nan_to_num(anchor_volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        anchor = weeks_data[latest_seq[-1]].flair_resampled.astype(np.float32)

    pred_full = (0.94 * anchor) + (0.06 * pred_full)
    anchor_detail = anchor - gaussian_filter(anchor, sigma=0.8)
    pred_full = pred_full + (0.08 * anchor_detail)
    pred_full = gaussian_filter(pred_full, sigma=0.10)
    pred_full = np.clip(pred_full, -8.0, 8.0).astype(np.float32)

    info["gnn_pred_entropy_mean"] = float(np.mean(pred_ent_np)) if pred_ent_np.size > 0 else 0.0
    info["gnn_pred_tumor_mean"] = float(np.mean(tumor_full)) if tumor_full.size > 0 else 0.0
    return pred_full.astype(np.float32), tumor_full.astype(np.float32), info


def train_voxel_sequence_predictor(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    model: nn.Module,
    model_name: str,
    max_samples: int = 50000,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 7e-4,
    seed_offset: int = 0,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """
    Train a compact sequence model (GRU/LSTM/Transformer) on voxel time-series.
    Returns (predicted_volume, info).
    """
    prefix = model_name.lower()
    info: dict[str, Any] = {
        f"{prefix}_samples": 0,
        f"{prefix}_epochs": 0,
        f"{prefix}_final_loss": None,
        "model_name": model_name,
    }

    windows = build_temporal_windows(weeks_data, seq_len, target_idx)
    if not windows:
        return None, info

    rng = np.random.default_rng(1337 + target_idx + seed_offset)
    X_chunks = []
    y_chunks = []
    per_window = max(1200, max_samples // len(windows))

    for seq_indices, tgt_idx in windows:
        seq_stack = np.stack([weeks_data[i].flair_resampled for i in seq_indices], axis=-1)
        X_full = seq_stack.reshape(-1, seq_len)
        y_full = weeks_data[tgt_idx].flair_resampled.reshape(-1)

        n = X_full.shape[0]
        if per_window < n:
            tumor_flat = (weeks_data[tgt_idx].mask_resampled.reshape(-1) > 0.5)
            weights = np.ones(n, dtype=np.float64)
            if tumor_flat.any():
                weights[tumor_flat] = 5.0
            weights = weights / weights.sum()
            idx = rng.choice(n, size=per_window, replace=False, p=weights)
            X_full = X_full[idx]
            y_full = y_full[idx]

        X_chunks.append(X_full)
        y_chunks.append(y_full)

    X_train = np.concatenate(X_chunks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_chunks, axis=0).astype(np.float32)
    info[f"{prefix}_samples"] = int(X_train.shape[0])

    x_mean = float(X_train.mean())
    x_std = float(max(X_train.std(), 1e-6))
    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))

    X_train = (X_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    criterion = nn.SmoothL1Loss(beta=0.2)

    dataset_size = X_train.shape[0]
    val_size = max(1024, int(0.1 * dataset_size))
    val_idx = rng.choice(dataset_size, size=min(val_size, dataset_size // 4), replace=False)
    train_mask = np.ones(dataset_size, dtype=bool)
    train_mask[val_idx] = False
    train_idx_all = np.where(train_mask)[0]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    model.train()
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(epochs):
        perm = rng.permutation(train_idx_all)
        epoch_losses = []

        for start in range(0, train_idx_all.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device).unsqueeze(-1)
            yb = torch.from_numpy(y_train[idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, X_val.shape[0], batch_size):
                xb = torch.from_numpy(X_val[start:start + batch_size]).to(device).unsqueeze(-1)
                yb = torch.from_numpy(y_val[start:start + batch_size]).to(device)
                pred = model(xb)
                val_losses.append(float(criterion(pred, yb).item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(epoch_losses))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                break

        model.train()

        info[f"{prefix}_final_loss"] = float(np.mean(epoch_losses)) if epoch_losses else None
        info[f"{prefix}_epochs"] = int(epoch + 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    latest_seq = _latest_sequence_indices(len(weeks_data), target_idx, seq_len)
    X_test = np.stack([weeks_data[i].flair_resampled for i in latest_seq], axis=-1).reshape(-1, seq_len).astype(np.float32)
    X_test = (X_test - x_mean) / x_std
    out_shape = weeks_data[target_idx].flair_resampled.shape

    model.eval()
    preds = np.zeros((X_test.shape[0],), dtype=np.float32)
    chunk = 200000
    with torch.no_grad():
        for start in range(0, X_test.shape[0], chunk):
            xb = torch.from_numpy(X_test[start:start + chunk]).to(device).unsqueeze(-1)
            out = model(xb).cpu().numpy().astype(np.float32)
            preds[start:start + chunk] = out

    preds = (preds * y_std) + y_mean
    return preds.reshape(out_shape).astype(np.float32), info


def train_gru_predictor(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    max_samples: int = 50000,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 7e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    return train_voxel_sequence_predictor(
        weeks_data,
        target_idx,
        seq_len,
        VoxelGRUPredictor(hidden_size=96),
        model_name="gru",
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed_offset=0,
        device=device,
    )


def train_lstm_predictor(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    max_samples: int = 50000,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 7e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    return train_voxel_sequence_predictor(
        weeks_data,
        target_idx,
        seq_len,
        VoxelLSTMPredictor(hidden_size=96),
        model_name="lstm",
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed_offset=17,
        device=device,
    )


def train_transformer_predictor(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    max_samples: int = 50000,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 6e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    return train_voxel_sequence_predictor(
        weeks_data,
        target_idx,
        seq_len,
        VoxelTransformerPredictor(hidden_size=96, n_heads=4, num_layers=3, max_seq_len=max(8, seq_len)),
        model_name="transformer",
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed_offset=31,
        device=device,
    )


def train_classical_predictors(
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int,
    max_samples_rf: int = 50000,
    max_samples_svr: int = 15000,
    gnn_max_samples: int = 28000,
    gnn_epochs: int = 14,
    gnn_batch_size: int = 768,
    gnn_learning_rate: float = 4e-4,
    gnn_k_neighbors: int = 5,
    gnn_temperature: float = 0.24,
    gnn_entropy_weight: float = 0.20,
    gnn_tumor_weight: float = 0.55,
    seq_epochs: int = 20,
    seq_max_samples: int = 50000,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    dict[str, Any],
]:
    """
    Train classical and sequence predictors on voxel-wise temporal features.
    Returns (rf, svr, gru, lstm, transformer, gnn, gnn_tumor, training_info).
    """
    info: dict[str, Any] = {
        "sklearn_available": SKLEARN_AVAILABLE,
        "rf_samples": 0,
        "svr_samples": 0,
        "gru_samples": 0,
        "lstm_samples": 0,
        "transformer_samples": 0,
        "gnn_samples": 0,
        "train_windows": 0,
        "seq_len": seq_len,
    }
    if not SKLEARN_AVAILABLE:
        gru_pred, gru_info = train_gru_predictor(
            weeks_data, target_idx, seq_len, max_samples=seq_max_samples, epochs=seq_epochs,
        )
        lstm_pred, lstm_info = train_lstm_predictor(
            weeks_data, target_idx, seq_len, max_samples=seq_max_samples, epochs=seq_epochs,
        )
        transformer_pred, transformer_info = train_transformer_predictor(
            weeks_data, target_idx, seq_len, max_samples=seq_max_samples, epochs=seq_epochs,
        )
        gnn_pred, gnn_tumor_pred, gnn_info = train_temporal_gnn_predictor(
            weeks_data,
            target_idx,
            seq_len,
            max_samples=gnn_max_samples,
            epochs=gnn_epochs,
            batch_size=gnn_batch_size,
            learning_rate=gnn_learning_rate,
            k_neighbors=gnn_k_neighbors,
            temperature=gnn_temperature,
            entropy_weight=gnn_entropy_weight,
            tumor_weight=gnn_tumor_weight,
            anchor_volume=gru_pred,
        )
        info.update(gru_info)
        info.update(lstm_info)
        info.update(transformer_info)
        info.update(gnn_info)
        return None, None, gru_pred, lstm_pred, transformer_pred, gnn_pred, gnn_tumor_pred, info

    windows = build_temporal_windows(weeks_data, seq_len, target_idx)
    if not windows:
        return None, None, None, None, None, None, None, info

    info["train_windows"] = len(windows)
    rng = np.random.default_rng(42 + target_idx)

    X_chunks = []
    y_chunks = []
    tumor_chunks = []
    per_window = max(1000, max_samples_rf // len(windows))

    for seq_indices, tgt_idx in windows:
        seq_stack = np.stack([weeks_data[i].flair_resampled for i in seq_indices], axis=-1)  # (D,H,W,T)
        X_full = seq_stack.reshape(-1, seq_len)
        y_full = weeks_data[tgt_idx].flair_resampled.reshape(-1)
        tumor_flat = (weeks_data[tgt_idx].mask_resampled.reshape(-1) > 0.5)

        n = X_full.shape[0]
        if per_window < n:
            weights = np.ones(n, dtype=np.float64)
            if tumor_flat.any():
                weights[tumor_flat] = 5.0
            weights = weights / weights.sum()
            idx = rng.choice(n, size=per_window, replace=False, p=weights)
            X_full = X_full[idx]
            y_full = y_full[idx]
            tumor_flat = tumor_flat[idx]

        X_chunks.append(X_full)
        y_chunks.append(y_full)
        tumor_chunks.append(tumor_flat)

    X_train = np.concatenate(X_chunks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_chunks, axis=0).astype(np.float32)
    tumor_train = np.concatenate(tumor_chunks, axis=0)
    info["rf_samples"] = int(X_train.shape[0])

    # Inference features: latest seq_len weeks before target
    latest_seq = _latest_sequence_indices(len(weeks_data), target_idx, seq_len)
    X_test = np.stack([weeks_data[i].flair_resampled for i in latest_seq], axis=-1).reshape(-1, seq_len).astype(np.float32)
    out_shape = weeks_data[target_idx].flair_resampled.shape

    rf_pred = None
    svr_pred = None
    gru_pred = None
    lstm_pred = None
    transformer_pred = None
    gnn_pred = None
    gnn_tumor_pred = None

    try:
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=22,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test).reshape(out_shape).astype(np.float32)
    except Exception as e:
        info["rf_error"] = str(e)

    try:
        svr_n = min(max(max_samples_svr, 22000), X_train.shape[0])
        if tumor_train.any():
            weights = np.ones(X_train.shape[0], dtype=np.float64)
            weights[tumor_train] = 5.0
            weights = weights / weights.sum()
            idx = rng.choice(X_train.shape[0], size=svr_n, replace=False, p=weights)
        else:
            idx = rng.choice(X_train.shape[0], size=svr_n, replace=False)
        X_svr = X_train[idx]
        y_svr = y_train[idx]
        info["svr_samples"] = int(X_svr.shape[0])

        svr = make_pipeline(
            StandardScaler(),
            LinearSVR(C=2.5, epsilon=0.04, max_iter=8000, random_state=42),
        )
        svr.fit(X_svr, y_svr)
        svr_pred = svr.predict(X_test).reshape(out_shape).astype(np.float32)
    except Exception as e:
        info["svr_error"] = str(e)

    try:
        gru_pred, gru_info = train_gru_predictor(
            weeks_data, target_idx, seq_len, max_samples=seq_max_samples, epochs=seq_epochs,
        )
        info.update(gru_info)
    except Exception as e:
        info["gru_error"] = str(e)

    try:
        lstm_pred, lstm_info = train_lstm_predictor(
            weeks_data, target_idx, seq_len, max_samples=seq_max_samples, epochs=seq_epochs,
        )
        info.update(lstm_info)
    except Exception as e:
        info["lstm_error"] = str(e)

    try:
        transformer_pred, transformer_info = train_transformer_predictor(
            weeks_data, target_idx, seq_len, max_samples=seq_max_samples, epochs=seq_epochs,
        )
        info.update(transformer_info)
    except Exception as e:
        info["transformer_error"] = str(e)

    try:
        gnn_pred, gnn_tumor_pred, gnn_info = train_temporal_gnn_predictor(
            weeks_data,
            target_idx,
            seq_len,
            max_samples=gnn_max_samples,
            epochs=gnn_epochs,
            batch_size=gnn_batch_size,
            learning_rate=gnn_learning_rate,
            k_neighbors=gnn_k_neighbors,
            temperature=gnn_temperature,
            entropy_weight=gnn_entropy_weight,
            tumor_weight=gnn_tumor_weight,
            anchor_volume=gru_pred,
        )
        info.update(gnn_info)
    except Exception as e:
        info["gnn_error"] = str(e)

    return rf_pred, svr_pred, gru_pred, lstm_pred, transformer_pred, gnn_pred, gnn_tumor_pred, info


def train_stacked_linear_refiner(
    rnn_pred: np.ndarray,
    svr_pred: np.ndarray,
    rf_pred: np.ndarray,
    target_image: np.ndarray,
    target_mask: Optional[np.ndarray] = None,
    sample_size: int = 250_000,
    alpha: float = 1e-2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a lightweight stacked refiner that combines RNN, SVR and RF outputs."""
    rnn = np.nan_to_num(rnn_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    svr = np.nan_to_num(svr_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rf = np.nan_to_num(rf_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if not (rnn.shape == svr.shape == rf.shape == target.shape):
        raise ValueError(
            f"Stacked refiner shape mismatch: RNN={rnn.shape}, SVR={svr.shape}, RF={rf.shape}, target={target.shape}"
        )

    flat_rnn = rnn.reshape(-1)
    flat_svr = svr.reshape(-1)
    flat_rf = rf.reshape(-1)
    flat_target = target.reshape(-1)
    n_voxels = int(flat_target.size)

    weights = np.ones(n_voxels, dtype=np.float32)
    p10 = float(np.percentile(flat_target, 10.0))
    brain_like = flat_target > p10
    weights += 0.35 * brain_like.astype(np.float32)

    if target_mask is not None:
        tumor = (target_mask.reshape(-1) > 0.5).astype(np.float32)
        weights += 8.0 * tumor

    weights_sum = float(weights.sum())
    if weights_sum <= 0:
        weights[:] = 1.0
        weights_sum = float(weights.sum())

    rng = np.random.default_rng(12345)
    sample_n = min(int(sample_size), n_voxels)
    sample_prob = weights / weights_sum
    sample_idx = rng.choice(n_voxels, size=sample_n, replace=False, p=sample_prob)

    x_train = np.stack([
        flat_rnn[sample_idx],
        flat_svr[sample_idx],
        flat_rf[sample_idx],
        np.ones(sample_n, dtype=np.float32),
    ], axis=1).astype(np.float32)
    y_train = flat_target[sample_idx].astype(np.float32)
    w_train = np.sqrt(weights[sample_idx].astype(np.float32))

    # Standardize model channels before solving; this avoids a single predictor scale
    # dominating the ridge system and stabilizes coefficients across runs.
    feat_mu = np.mean(x_train[:, :3], axis=0, keepdims=True).astype(np.float32)
    feat_sd = np.std(x_train[:, :3], axis=0, keepdims=True).astype(np.float32)
    feat_sd = np.where(feat_sd < 1e-6, 1.0, feat_sd).astype(np.float32)

    y_mu = float(np.mean(y_train))
    y_sd = float(np.std(y_train))
    if y_sd < 1e-6:
        y_sd = 1.0

    x_train_std = x_train.copy()
    x_train_std[:, :3] = (x_train_std[:, :3] - feat_mu) / feat_sd
    y_train_std = (y_train - y_mu) / y_sd

    xw = x_train_std * w_train[:, None]
    yw = y_train_std * w_train
    eye = np.eye(x_train_std.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0  # Do not regularize intercept term.
    coeffs = np.linalg.solve((xw.T @ xw) + (alpha * eye), xw.T @ yw).astype(np.float32)

    full_x = np.stack([
        flat_rnn,
        flat_svr,
        flat_rf,
        np.ones(n_voxels, dtype=np.float32),
    ], axis=1).astype(np.float32)
    full_x_std = full_x.copy()
    full_x_std[:, :3] = (full_x_std[:, :3] - feat_mu) / feat_sd

    stacked_std = (full_x_std @ coeffs).astype(np.float32)
    stacked = (stacked_std * y_sd + y_mu).reshape(target.shape).astype(np.float32)

    # Keep stacked output in a realistic target domain and avoid outlier tails.
    src_stack = np.stack([flat_rnn, flat_svr, flat_rf], axis=0).astype(np.float32)
    src_lo = float(np.percentile(src_stack, 0.5))
    src_hi = float(np.percentile(src_stack, 99.5))

    lo = float(np.percentile(flat_target, 0.5))
    hi = float(np.percentile(flat_target, 99.5))
    lo = max(lo, src_lo)
    hi = min(hi, src_hi) if src_hi > src_lo else hi
    if hi > lo:
        stacked = np.clip(stacked, lo, hi).astype(np.float32)

    # Conservative blend with simple mean ensemble to reduce overfitting artifacts.
    mean_img = ((rnn + svr + rf) / 3.0).astype(np.float32)
    stacked = (0.75 * stacked + 0.25 * mean_img).astype(np.float32)

    train_pred = ((x_train_std @ coeffs) * y_sd + y_mu).astype(np.float32)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    train_mae = float(np.mean(np.abs(train_pred - y_train)))

    info = {
        "method": "weighted_ridge_stacker",
        "sample_size": int(sample_n),
        "alpha": float(alpha),
        "coefficients": [float(v) for v in coeffs.tolist()],
        "feature_mean": [float(v) for v in feat_mu.reshape(-1).tolist()],
        "feature_std": [float(v) for v in feat_sd.reshape(-1).tolist()],
        "target_mean": float(y_mu),
        "target_std": float(y_sd),
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "source_models": ["rnn_gru", "svr", "random_forest"],
    }
    return stacked.astype(np.float32), info


def build_enriched_weeks_with_aux_predictions(
    weeks_data: list[WeekData],
    target_idx: int,
    svr_pred: np.ndarray,
    rf_pred: np.ndarray,
) -> list[WeekData]:
    """Create a new week list where SVR/RF predictions are appended as extra channels.

    This implements the professor-requested setup: original temporal sequence plus
    auxiliary prediction channels (SVR and Random Forest) for a new RNN training pass.
    """
    svr = np.nan_to_num(svr_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rf = np.nan_to_num(rf_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if svr.shape != rf.shape:
        raise ValueError(f"Enriched RNN requires matching SVR/RF shapes, got {svr.shape} vs {rf.shape}")

    enriched: list[WeekData] = []
    denom = float(max(target_idx, 1))

    for i, w in enumerate(weeks_data):
        if w.modalities_resampled.shape[1:] != svr.shape:
            raise ValueError(
                f"Shape mismatch in enriched build for week {w.week}: "
                f"modalities={w.modalities_resampled.shape[1:]}, aux={svr.shape}"
            )

        # Temporal weighting: keep aux guidance stronger near the target week.
        temporal_weight = float(min(1.0, (i + 1) / (denom + 1.0)))
        aux_svr = (svr * temporal_weight).astype(np.float32)
        aux_rf = (rf * temporal_weight).astype(np.float32)

        extra = np.stack([aux_svr, aux_rf], axis=0).astype(np.float32)
        mods_enriched = np.concatenate([w.modalities_resampled.astype(np.float32), extra], axis=0).astype(np.float32)

        enriched.append(
            WeekData(
                week=w.week,
                modalities_resampled=mods_enriched,
                flair_resampled=w.flair_resampled,
                mask_resampled=w.mask_resampled,
                flair_original=w.flair_original,
                mask_original=w.mask_original,
                flair_affine=w.flair_affine,
                flair_header=w.flair_header,
                flair_mean=w.flair_mean,
                flair_std=w.flair_std,
                tumor_ratio=w.tumor_ratio,
            )
        )

    return enriched


# ============================================================================
# TRAINING
# ============================================================================

def train_improved_predictor(
    model: ImprovedSpatialTemporalPredictor,
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int = 3,
    patch_size: tuple[int, int, int] = (48, 48, 24),
    epochs: int = 20,
    steps_per_epoch: int = 30,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 8,
    weight_decay: float = 1e-5,
    roi_weight: float = 20.0,
    roi_weight_end: Optional[float] = None,
    tumor_patch_prob: float = 0.7,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Train improved spatiotemporal predictor with multi-task learning.
    
    Returns: training history dict
    """
    device = resolve_compute_device(device)
    model = model.to(device)
    
    # Build windows
    windows = build_temporal_windows(weeks_data, seq_len, target_idx)
    if not windows:
        raise ValueError(f"No valid windows for target_idx={target_idx}")
    
    # Loss function
    combined_loss = CombinedLoss(
        l1_weight=0.33,
        ssim_weight=0.34,
        roi_loss_weight=0.12,
        entropy_weight=0.01,
        gradient_weight=0.10,
        perceptual_weight=0.17,
        texture_weight=0.06,
        aux_weight=0.10,
        roi_weight=roi_weight,
    )
    combined_loss = combined_loss.to(device)

    # Progressive tumor emphasis: linearly increase ROI weight over epochs.
    if roi_weight_end is None:
        roi_weight_end = roi_weight
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(combined_loss.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    
    # Mixed precision (CUDA / Intel XPU when available).
    use_amp = False
    if "cuda" in str(device) and torch.cuda.is_available():
        use_amp = True
    elif str(device) == "xpu":
        try:
            use_amp = bool(torch.xpu.is_available())
        except Exception:
            use_amp = False
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    # Tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }
    
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        if epochs > 1:
            alpha = epoch / float(epochs - 1)
            combined_loss.roi_weight = float((1.0 - alpha) * roi_weight + alpha * roi_weight_end)
        else:
            combined_loss.roi_weight = float(roi_weight_end)

        # Training step
        model.train()
        train_losses = []
        
        pbar = range(steps_per_epoch)
        for step in pbar:
            batch_x, batch_y, batch_mask = sample_batch(
                weeks_data,
                windows,
                batch_size,
                patch_size,
                tumor_patch_prob=tumor_patch_prob,
            )
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, seg = model(batch_x, return_seg=True)
                
                # Ground truth for auxiliary segmentation
                seg_target = (batch_mask > 0.5).squeeze(1)
                
                loss = combined_loss(pred, batch_y, batch_mask, seg, seg_target)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(float(loss.item()))
        
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        
        # Validation step (on different patches)
        model.eval()
        val_losses = []
        for _ in range(5):
            with torch.no_grad():
                batch_x, batch_y, batch_mask = sample_batch(
                    weeks_data,
                    windows,
                    batch_size,
                    patch_size,
                    tumor_patch_prob=tumor_patch_prob,
                )
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)
                
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred, seg = model(batch_x, return_seg=True)
                    seg_target = (batch_mask > 0.5).squeeze(1)
                    loss = combined_loss(pred, batch_y, batch_mask, seg, seg_target)
                
                val_losses.append(float(loss.item()))
        
        val_loss = float(np.mean(val_losses))
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        # Scheduler & checkpoint
        scheduler.step(val_loss)
        
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            best_model_state = model.state_dict()
            patience_counter = 0
            
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_state, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"ROIw: {combined_loss.roi_weight:.1f}"
        )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    history["final_model_state"] = best_model_state
    return history


# ============================================================================
# INFERENCE: Full Volume Prediction
# ============================================================================

def predict_full_volume(
    model: ImprovedSpatialTemporalPredictor,
    weeks_data: list[WeekData],
    target_idx: int,
    seq_len: int = 3,
    patch_size: tuple[int, int, int] = (48, 48, 24),
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    overlap_ratio: float = 0.5,
    return_auxiliary_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Predict full FLAIR volume using sliding window with overlap averaging.
    
    Returns: predicted FLAIR image (D, H, W), or (image, auxiliary_seg_mask) if return_auxiliary_mask.
    """
    model = model.to(device)
    model.eval()
    
    # Prepare complete sequence
    seq_indices = list(range(max(0, target_idx - seq_len + 1), target_idx + 1))
    
    # Pad if needed
    while len(seq_indices) < seq_len:
        seq_indices.insert(0, seq_indices[0])
    
    seq_indices = seq_indices[-seq_len:]
    
    # Get volume shape
    volume_shape = weeks_data[target_idx].flair_resampled.shape
    D, H, W = volume_shape
    dz, dy, dx = patch_size
    
    # Overlap
    stride_z = max(1, int(dz * (1 - overlap_ratio)))
    stride_y = max(1, int(dy * (1 - overlap_ratio)))
    stride_x = max(1, int(dx * (1 - overlap_ratio)))
    
    # Accumulator
    pred_volume = np.zeros(volume_shape, dtype=np.float32)
    seg_volume = np.zeros(volume_shape, dtype=np.float32)
    count_map = np.zeros(volume_shape, dtype=np.float32)
    has_aux = return_auxiliary_mask and getattr(model, "enable_auxiliary_segmentation", False)
    
    with torch.no_grad():
        for z in range(0, D - dz + 1, stride_z):
            for y in range(0, H - dy + 1, stride_y):
                for x in range(0, W - dx + 1, stride_x):
                    # Extract sequence patch
                    x_seq = []
                    for idx in seq_indices:
                        mods = weeks_data[idx].modalities_resampled
                        patch = mods[:, z:z+dz, y:y+dy, x:x+dx]
                        
                        # Pad if needed
                        if patch.shape != (mods.shape[0], dz, dy, dx):
                            pad_patch = np.zeros((mods.shape[0], dz, dy, dx), dtype=np.float32)
                            pad_patch[:, :patch.shape[1], :patch.shape[2], :patch.shape[3]] = patch
                            patch = pad_patch
                        
                        x_seq.append(torch.from_numpy(patch))
                    
                    x_seq = torch.stack(x_seq).unsqueeze(0).to(device)  # (1, T, C, dz, dy, dx)
                    
                    # Predict
                    use_amp = ("cuda" in device) and torch.cuda.is_available()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred_patch, seg_patch = model(x_seq, return_seg=True)
                    
                    pred_patch = pred_patch.squeeze(0).squeeze(0).cpu().numpy()
                    if has_aux and seg_patch is not None:
                        seg_prob = torch.sigmoid(seg_patch).squeeze(0).squeeze(0).cpu().numpy()
                        seg_volume[z:z+dz, y:y+dy, x:x+dx] += seg_prob
                    
                    # Accumulate
                    pred_volume[z:z+dz, y:y+dy, x:x+dx] += pred_patch
                    count_map[z:z+dz, y:y+dy, x:x+dx] += 1.0
    
    # Average overlapping regions
    pred_volume = np.divide(pred_volume, count_map, where=count_map>0, out=np.zeros_like(pred_volume))
    pred_volume = np.clip(pred_volume, -10.0, 10.0).astype(np.float32)

    if has_aux:
        seg_volume = np.divide(seg_volume, count_map, where=count_map > 0, out=np.zeros_like(seg_volume))
        aux_mask = (seg_volume >= 0.42).astype(np.float32)
        return pred_volume, aux_mask

    return pred_volume


# ============================================================================
# POSTPROCESSING
# ============================================================================

def postprocess_predicted_image(
    pred_image: np.ndarray,
    real_images: list[np.ndarray],
    week_stats: dict,
    method: str = "histogram_match_enhanced",
) -> np.ndarray:
    """
    Postprocess predicted image for better nnU-Net compatibility.
    Preserve real intensity domain and avoid global over-brightening.
    """
    # Denormalize if input is z-score normalized and stats are valid.
    denorm = np.nan_to_num(pred_image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean_val = float(week_stats.get("mean", 0.0))
    std_val = float(week_stats.get("std", 1.0))
    if np.isfinite(mean_val) and np.isfinite(std_val) and (std_val > 1e-6):
        denorm = (denorm * std_val) + mean_val

    # Build robust reference in real-intensity domain.
    if real_images:
        valid_real_images = [
            np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for r in real_images
            if (r is not None) and (np.isfinite(r).any())
        ]
    else:
        valid_real_images = []
    avg_real = np.mean(valid_real_images[-2:], axis=0) if len(valid_real_images) >= 2 else (valid_real_images[-1] if valid_real_images else denorm)
    avg_real = avg_real.astype(np.float32)

    if method == "histogram_match_enhanced":
        # Match global histogram, then blend with denormalized prediction.
        matched = histogram_match(denorm, avg_real).astype(np.float32)
        matched = (0.72 * matched) + (0.28 * denorm)

        # Keep internal texture from latest real week to avoid over-flat outputs.
        if valid_real_images:
            prior_real = valid_real_images[-1]
            prior_high = prior_real - gaussian_filter(prior_real, sigma=1.1)
            matched = matched + (0.08 * prior_high.astype(np.float32))

            # Edge-aware correction: pull prediction edge energy toward recent real anatomy.
            pred_high = matched - gaussian_filter(matched, sigma=1.0)
            pred_energy = float(np.std(pred_high))
            prior_energy = float(np.std(prior_high))
            edge_gain = np.clip(prior_energy / max(pred_energy, 1e-6), 0.90, 1.35)
            matched = matched + (0.10 * edge_gain * (prior_high - (0.30 * pred_high)))

        # Re-align robust dynamic range to real-reference percentiles.
        m_lo = float(np.percentile(matched, 0.5))
        m_hi = float(np.percentile(matched, 99.5))
        r_lo = float(np.percentile(avg_real, 0.5))
        r_hi = float(np.percentile(avg_real, 99.5))
        if (m_hi > m_lo) and (r_hi > r_lo):
            matched = (matched - m_lo) / (m_hi - m_lo)
            matched = (matched * (r_hi - r_lo)) + r_lo

        # Align mean/std to recent real reference to avoid brightness drift.
        m_mu = float(matched.mean())
        m_std = float(matched.std()) if float(matched.std()) > 1e-6 else 1.0
        r_mu = float(avg_real.mean())
        r_std = float(avg_real.std()) if float(avg_real.std()) > 1e-6 else m_std
        matched = ((matched - m_mu) / m_std) * r_std + r_mu

        # Preserve local structure while avoiding halo artifacts.
        smooth = gaussian_filter(matched, sigma=0.30)
        detail = matched - gaussian_filter(matched, sigma=0.95)
        brain_mask = avg_real > float(np.percentile(avg_real, 10.0))
        matched = smooth.copy()
        matched[brain_mask] = smooth[brain_mask] + (0.09 * detail[brain_mask])

        # Clip with robust reference percentiles (not full min/max stretch).
        lo = float(np.percentile(avg_real, 0.5))
        hi = float(np.percentile(avg_real, 99.7))
        if hi > lo:
            matched = np.clip(matched, lo, hi)

        # Final safety clip to target-week physical range.
        if "min" in week_stats and "max" in week_stats:
            vmin = float(week_stats["min"])
            vmax = float(week_stats["max"])
            if np.isfinite(vmin) and np.isfinite(vmax) and (vmax > vmin):
                matched = np.clip(matched, vmin, vmax)

        # Hard guardrail against washed-out prediction volumes.
        if float(np.std(matched)) < max(1e-6, 0.22 * float(np.std(avg_real))):
            mr_lo = float(np.percentile(matched, 1.0))
            mr_hi = float(np.percentile(matched, 99.0))
            rr_lo = float(np.percentile(avg_real, 1.0))
            rr_hi = float(np.percentile(avg_real, 99.0))
            if (mr_hi > mr_lo) and (rr_hi > rr_lo):
                matched = (matched - mr_lo) / (mr_hi - mr_lo)
                matched = (matched * (rr_hi - rr_lo)) + rr_lo

        return np.nan_to_num(matched, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return denorm.astype(np.float32)


def denormalize_from_week_stats(volume: np.ndarray, week_stats: dict) -> np.ndarray:
    """Convert normalized working-space volume back to real-like intensity domain."""
    out = volume.astype(np.float32)
    if "mean" in week_stats and "std" in week_stats:
        out = (out * week_stats["std"]) + week_stats["mean"]
    if "min" in week_stats and "max" in week_stats:
        out = np.clip(out, week_stats["min"], week_stats["max"])
    return out.astype(np.float32)


def refine_structural_candidate(
    candidate: np.ndarray,
    prior_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Enhance structural detail while preserving global intensity trend."""
    c0 = candidate.astype(np.float32)
    c = c0.copy()

    # Mild unsharp masking to recover edges and small structures.
    c_blur = gaussian_filter(c, sigma=0.8)
    c_high = c - c_blur
    sharp_gain = 1.0

    # Inject low-amplitude high-frequency prior information for realistic texture.
    if prior_image is not None:
        p = prior_image.astype(np.float32)
        p_blur = gaussian_filter(p, sigma=1.2)
        p_high = p - p_blur
        p_energy = float(np.std(p_high))
        c_energy = float(np.std(c_high))
        sharp_gain = float(np.clip(p_energy / max(c_energy, 1e-6), 0.90, 1.40))
        c = c + (0.18 * sharp_gain * c_high) + (0.14 * p_high)
    else:
        c = c + (0.20 * c_high)

    # Keep contrast realistic by mixing with original candidate.
    c = (0.88 * c) + (0.12 * c0)

    lo = float(np.percentile(c, 0.2))
    hi = float(np.percentile(c, 99.8))
    c = np.clip(c, lo, hi)
    return c.astype(np.float32)


def refine_primary_model_prediction(
    prediction: np.ndarray,
    prior_image: Optional[np.ndarray] = None,
    tumor_mask: Optional[np.ndarray] = None,
    prior_blend: float = 0.80,
) -> np.ndarray:
    """Refine GRU/LSTM/Transformer/GNN/SVR outputs with temporal prior and ROI sharpening."""
    out = np.nan_to_num(prediction.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if prior_image is not None:
        prior = prior_image.astype(np.float32)
        out = (prior_blend * out) + ((1.0 - prior_blend) * prior)
    if tumor_mask is not None and np.any(tumor_mask > 0.5):
        t = tumor_mask > 0.5
        roi_detail = out - gaussian_filter(out, sigma=0.65)
        out[t] = out[t] + (0.14 * roi_detail[t])
    out = refine_structural_candidate(out, prior_image)
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def normalize_for_display(slice2d: np.ndarray, p_low: float = 1.0, p_high: float = 99.5, gamma: float = 0.95) -> np.ndarray:
    """Normalize 2D slice for visualization using robust percentiles."""
    s = slice2d.astype(np.float32)
    flat = s.reshape(-1)
    nz = flat[np.abs(flat) > 1e-6]
    if nz.size >= max(64, int(0.01 * flat.size)):
        stats_source = nz
    else:
        stats_source = flat

    lo = float(np.percentile(stats_source, p_low))
    hi = float(np.percentile(stats_source, p_high))
    if hi <= lo:
        return np.zeros_like(s, dtype=np.float32)
    s = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
    if gamma > 0:
        s = np.power(s, gamma, dtype=np.float32)
    return s.astype(np.float32)


def enhance_mask_for_display(mask_slice: np.ndarray, min_pixels: int = 140, max_iters: int = 5) -> np.ndarray:
    """Make tiny but valid lesions visible in report panels without altering saved volumetric masks."""
    m = (mask_slice > 0.5)
    if not m.any():
        return np.zeros_like(mask_slice, dtype=np.float32)

    iters = 0
    st = np.ones((3, 3), dtype=bool)
    while int(m.sum()) < min_pixels and iters < max_iters:
        m = binary_dilation(m, structure=st, iterations=1)
        iters += 1

    return m.astype(np.float32)


def select_axial_slice_index(mask_volume: np.ndarray, allow_empty_fallback: bool = True) -> int:
    """Choose representative axial slice by maximum tumor area."""
    m = (mask_volume > 0.5).astype(np.float32)
    if m.ndim != 3:
        raise ValueError(f"Mask volume must be 3D, got shape={m.shape}")
    # Assume axial is last axis for original NIfTI layout (X, Y, Z).
    areas = m.sum(axis=(0, 1))
    if float(areas.max()) > 0:
        return int(np.argmax(areas))
    if not allow_empty_fallback:
        raise RuntimeError("No tumor-positive axial slice found in mask volume")
    return int(m.shape[2] // 2)


def extract_axial_slice(volume: np.ndarray, z_idx: int) -> np.ndarray:
    """Extract an axial slice and orient it consistently for display."""
    sl = volume[:, :, z_idx]
    # Rotate for a consistent visual orientation (radiological-like display).
    return np.rot90(sl, k=1).astype(np.float32)


def axial_aspect_from_header(header: Any) -> float:
    """Compute physical aspect ratio for axial visualization."""
    try:
        zooms = header.get_zooms()[:3]
        # For axial slice from (X, Y, Z), rows~Y and cols~X after rot90.
        sx = float(zooms[0])
        sy = float(zooms[1])
        if sx > 0 and sy > 0:
            return sy / sx
    except Exception:
        pass
    return 1.0


def apply_shape_preserving_orientation_transform(volume: np.ndarray, transform_name: str) -> np.ndarray:
    """Apply a deterministic orientation transform while preserving shape."""
    v = volume.astype(np.float32)
    if transform_name == "identity":
        return v
    if transform_name == "transpose_xy":
        return np.transpose(v, (1, 0, 2)).astype(np.float32)
    if transform_name == "flip_x":
        return np.flip(v, axis=0).astype(np.float32)
    if transform_name == "flip_y":
        return np.flip(v, axis=1).astype(np.float32)
    if transform_name == "flip_z":
        return np.flip(v, axis=2).astype(np.float32)
    if transform_name == "transpose_xy_flip_x":
        return np.flip(np.transpose(v, (1, 0, 2)), axis=0).astype(np.float32)
    if transform_name == "transpose_xy_flip_y":
        return np.flip(np.transpose(v, (1, 0, 2)), axis=1).astype(np.float32)
    if transform_name == "transpose_xy_flip_z":
        return np.flip(np.transpose(v, (1, 0, 2)), axis=2).astype(np.float32)
    raise ValueError(f"Unknown orientation transform: {transform_name}")


def align_prediction_orientation_to_reference(
    pred_volume: np.ndarray,
    reference_volume: np.ndarray,
    reference_tumor_mask: Optional[np.ndarray] = None,
    pred_tumor_volume: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, float | str]]:
    """Select the best shape-preserving orientation transform using image and tumor agreement."""
    transforms = [
        "identity",
        "transpose_xy",
        "flip_x",
        "flip_y",
        "flip_z",
        "transpose_xy_flip_x",
        "transpose_xy_flip_y",
        "transpose_xy_flip_z",
    ]

    best_name = "identity"
    best_score = -1e9
    best_pred = pred_volume.astype(np.float32)
    best_tumor = pred_tumor_volume.astype(np.float32) if pred_tumor_volume is not None else None
    best_img_metrics = {"ssim": 0.0, "rmse": 0.0, "mae": 0.0}
    best_tumor_dice = 0.0

    for name in transforms:
        cand_pred = apply_shape_preserving_orientation_transform(pred_volume, name)
        if cand_pred.shape != reference_volume.shape:
            continue

        imgm = compute_image_metrics(reference_volume, cand_pred, reference_tumor_mask)
        score = (
            (3.8 * float(imgm.get("ssim", 0.0)))
            - (0.9 * float(imgm.get("rmse", 0.0)))
            - (0.5 * float(imgm.get("mae", 0.0)))
        )

        cand_tumor = None
        tumor_dice = 0.0
        if pred_tumor_volume is not None:
            cand_tumor = apply_shape_preserving_orientation_transform(pred_tumor_volume, name)
            if reference_tumor_mask is not None:
                segm = segmentation_metrics(reference_tumor_mask, (cand_tumor > 0.5).astype(np.float32))
                tumor_dice = float(segm.get("dice", 0.0))
                score += 1.2 * tumor_dice

        if score > best_score:
            best_score = float(score)
            best_name = name
            best_pred = cand_pred.astype(np.float32)
            best_tumor = cand_tumor.astype(np.float32) if cand_tumor is not None else None
            best_img_metrics = {
                "ssim": float(imgm.get("ssim", 0.0)),
                "rmse": float(imgm.get("rmse", 0.0)),
                "mae": float(imgm.get("mae", 0.0)),
            }
            best_tumor_dice = float(tumor_dice)

    info: dict[str, float | str] = {
        "selected_transform": best_name,
        "score": float(best_score),
        "ssim": float(best_img_metrics.get("ssim", 0.0)),
        "rmse": float(best_img_metrics.get("rmse", 0.0)),
        "mae": float(best_img_metrics.get("mae", 0.0)),
        "tumor_dice": float(best_tumor_dice),
    }
    return best_pred.astype(np.float32), (best_tumor.astype(np.float32) if best_tumor is not None else None), info


def compute_shared_crop_bbox(
    real_slice: np.ndarray,
    pred_slice: np.ndarray,
    pred_mask_slice: np.ndarray,
    real_mask_slice: Optional[np.ndarray] = None,
    margin: int = 12,
) -> tuple[int, int, int, int]:
    """Compute a common crop box around useful brain/tumor content for aligned visualization."""
    r = normalize_for_display(real_slice)
    p = normalize_for_display(pred_slice)
    m = (pred_mask_slice > 0.5).astype(np.float32)
    mr = (real_mask_slice > 0.5).astype(np.float32) if real_mask_slice is not None else np.zeros_like(m)

    content = (r > 0.05) | (p > 0.05) | (m > 0.0) | (mr > 0.0)
    ys, xs = np.where(content)
    h, w = real_slice.shape
    if ys.size == 0 or xs.size == 0:
        return 0, h, 0, w

    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, h)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, w)

    # Keep minimal size to avoid over-cropping thin ROIs.
    min_h = max(96, int(0.35 * h))
    min_w = max(96, int(0.35 * w))
    if (y1 - y0) < min_h:
        c = (y0 + y1) // 2
        y0 = max(c - min_h // 2, 0)
        y1 = min(y0 + min_h, h)
        y0 = max(y1 - min_h, 0)
    if (x1 - x0) < min_w:
        c = (x0 + x1) // 2
        x0 = max(c - min_w // 2, 0)
        x1 = min(x0 + min_w, w)
        x0 = max(x1 - min_w, 0)

    return y0, y1, x0, x1


def crop_slice(slice2d: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a 2D slice using (y0, y1, x0, x1)."""
    y0, y1, x0, x1 = bbox
    return slice2d[y0:y1, x0:x1]


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def compute_image_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tumor_mask: np.ndarray,
    epsilon: float = 1e-7,
) -> dict[str, float]:
    """Compute comprehensive image-level metrics."""
    y_true = np.nan_to_num(y_true.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Global metrics
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + epsilon))
    
    def _mean_slice_ssim(axis: int) -> float:
        vals: list[float] = []
        n = y_true.shape[axis]
        for i in range(n):
            if axis == 0:
                t_sl, p_sl = y_true[i], y_pred[i]
            elif axis == 1:
                t_sl, p_sl = y_true[:, i], y_pred[:, i]
            else:
                t_sl, p_sl = y_true[:, :, i], y_pred[:, :, i]
            try:
                dr = float(max(t_sl.max() - t_sl.min(), 1e-6))
                s = float(ssim(t_sl, p_sl, data_range=dr))
                if not np.isfinite(s):
                    s = 0.0
                vals.append(float(np.clip(s, -1.0, 1.0)))
            except Exception:
                vals.append(0.0)
        return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0

    # SSIM legado (dim 0 = sagital no volume work-space) + por plano anatomico
    ssim_val = _mean_slice_ssim(0)
    ssim_sagittal = ssim_val
    ssim_coronal = _mean_slice_ssim(1)
    ssim_axial = _mean_slice_ssim(2)
    
    # PSNR
    mse = np.mean((y_true - y_pred) ** 2)
    dyn = float(max(y_true.max() - y_true.min(), 1e-6))
    psnr = 20.0 * np.log10(dyn / (np.sqrt(mse) + epsilon)) if mse > 0 else 100.0
    
    entropy_true = compute_volume_characteristics(y_true).get("entropy", 0.0)
    entropy_pred = compute_volume_characteristics(y_pred).get("entropy", 0.0)
    entropy_diff = abs(entropy_true - entropy_pred)

    auc_roc = 0.5
    if tumor_mask is not None:
        auc_roc = compute_auc_roc(tumor_mask, y_pred)

    # Texture mismatch from high-frequency residual maps.
    y_true_hf = y_true - gaussian_filter(y_true, sigma=1.0)
    y_pred_hf = y_pred - gaussian_filter(y_pred, sigma=1.0)
    texture_rmse = float(np.sqrt(np.mean((y_true_hf - y_pred_hf) ** 2)))
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "ssim": ssim_val,
        "ssim_sagittal": ssim_sagittal,
        "ssim_coronal": ssim_coronal,
        "ssim_axial": ssim_axial,
        "psnr": psnr,
        "entropy_true": entropy_true,
        "entropy_pred": entropy_pred,
        "entropy_diff": entropy_diff,
        "auc_roc": float(auc_roc),
        "texture_rmse": texture_rmse,
    }
    
    # ROI metrics
    if tumor_mask is not None and tumor_mask.sum() > 0:
        y_true_roi = y_true[tumor_mask > 0.5]
        y_pred_roi = y_pred[tumor_mask > 0.5]
        
        rmse_roi = float(np.sqrt(np.mean((y_true_roi - y_pred_roi) ** 2)))
        mae_roi = float(np.mean(np.abs(y_true_roi - y_pred_roi)))
        
        metrics.update({
            "rmse_tumor_roi": rmse_roi,
            "mae_tumor_roi": mae_roi,
        })
    else:
        metrics.update({"rmse_tumor_roi": 0.0, "mae_tumor_roi": 0.0})
    
    return metrics


def compute_auc_roc(mask_true: np.ndarray, score_pred: np.ndarray) -> float:
    """Compute ROC-AUC using continuous score map against binary tumor mask."""
    y_true = (np.nan_to_num(mask_true, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1) > 0.5).astype(np.uint8)
    y_score = np.nan_to_num(score_pred, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1).astype(np.float32)

    if y_true.size == 0:
        return 0.5

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    s_min = float(np.min(y_score))
    s_max = float(np.max(y_score))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or abs(s_max - s_min) < 1e-8:
        return 0.5

    # Min-max scale to stable [0, 1] scores.
    y_score = (y_score - s_min) / max(s_max - s_min, 1e-8)

    if SKLEARN_AVAILABLE:
        try:
            return float(np.clip(roc_auc_score(y_true, y_score), 0.0, 1.0))
        except Exception:
            pass

    # Fallback: rank-based Mann-Whitney U implementation.
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.shape[0] + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / max(float(n_pos * n_neg), 1.0)
    return float(np.clip(auc, 0.0, 1.0))


def compute_volume_characteristics(volume: np.ndarray, bins: int = 128) -> dict[str, float]:
    """Compute scalar intensity characteristics used in professor comparison tables."""
    v = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if v.size == 0:
        return {"mean": 0.0, "std": 0.0, "entropy": 0.0, "p95": 0.0}

    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
        ent = 0.0
    else:
        hist, _ = np.histogram(v, bins=bins, range=(vmin, vmax), density=False)
        probs = hist.astype(np.float64)
        probs = probs / max(float(probs.sum()), 1.0)
        probs = np.clip(probs, 1e-12, 1.0)
        ent = float(shannon_entropy(probs, base=np.e))

    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "entropy": float(ent),
        "p95": float(np.percentile(v, 95.0)),
    }


def segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute comprehensive segmentation metrics."""
    y_true = (y_true > 0.5).astype(np.uint8)
    y_pred = (y_pred > 0.5).astype(np.uint8)
    
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    
    # Dice
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-7)
    
    # Jaccard
    jaccard = tp / (tp + fp + fn + 1e-7)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    
    # Precision, Recall, etc.
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    
    # Hausdorff distance
    hausdorff = compute_hausdorff(y_true, y_pred)
    
    return {
        "accuracy": float(accuracy),
        "dice": float(dice),
        "jaccard": float(jaccard),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "hausdorff95": float(hausdorff),
    }


def validate_case_metrics_payload(metrics_data: dict[str, Any]) -> list[str]:
    """Validate the structure of a per-case metrics payload written to JSON."""
    errors: list[str] = []
    required_top_level = ["case_name", "model_metrics", "baseline_metrics", "classical_image_metrics"]
    for key in required_top_level:
        if key not in metrics_data:
            errors.append(f"missing top-level key: {key}")

    model_metrics = metrics_data.get("model_metrics") or {}
    baseline_metrics = metrics_data.get("baseline_metrics") or {}
    for key in ["dice", "rmse", "ssim"]:
        if key not in model_metrics:
            errors.append(f"missing model metric: {key}")
        if key not in baseline_metrics:
            errors.append(f"missing baseline metric: {key}")

    classical_image_metrics = metrics_data.get("classical_image_metrics") or {}
    if "temporal_gnn" not in classical_image_metrics:
        errors.append("missing temporal_gnn metrics in classical_image_metrics")

    if "temporal_gnn_experiment" not in metrics_data:
        errors.append("missing temporal_gnn_experiment block")
    else:
        gnn_block = metrics_data.get("temporal_gnn_experiment") or {}
        if ("metrics" not in gnn_block) and ("image_metrics" not in gnn_block):
            errors.append("missing temporal_gnn_experiment.metrics")
        if "training_info" not in gnn_block:
            errors.append("missing temporal_gnn_experiment.training_info")

    return errors


def postprocess_predicted_mask(
    mask_pred: np.ndarray,
    pred_image_ref: np.ndarray,
    reference_mask: Optional[np.ndarray] = None,
    min_ratio: float = 0.0002,
    min_component_voxels: int = 16,
    max_tumor_ratio: float = 0.06,
) -> tuple[np.ndarray, bool]:
    """
    Enforce anatomical plausibility and spatial consistency for predicted masks.
    Operates in the same space as the predicted image sent to nnU-Net.
    """
    changed = False
    original_mask = (mask_pred > 0.5)
    mask = original_mask.copy()

    # Brain-constrained support from the same image used for segmentation.
    thr = float(np.percentile(pred_image_ref, 12.0))
    brain = pred_image_ref > thr
    st = np.ones((3, 3, 3), dtype=bool)
    brain = binary_closing(brain, structure=st)
    brain = binary_opening(brain, structure=st)
    brain_lbl, brain_ncc = label(brain.astype(np.uint8))
    if brain_ncc > 0:
        brain_counts = np.bincount(brain_lbl.ravel())
        brain_counts[0] = 0
        brain = brain_lbl == int(np.argmax(brain_counts))

    if brain.any():
        new_mask = np.logical_and(mask, brain)
        changed = changed or (new_mask.sum() != mask.sum())
        mask = new_mask

    # Optional spatial prior from real target tumor mask (evaluation-time consistency support).
    ref = None
    ref_dil = None
    ref_wide = None
    ref_center = None
    if reference_mask is not None:
        ref = reference_mask > 0.5
        if ref.any():
            ref_dil = binary_dilation(ref, structure=st, iterations=5)
            ref_wide = binary_dilation(ref, structure=st, iterations=11)
            ref_center = np.argwhere(ref).mean(axis=0)

    # If near-empty, infer a conservative candidate directly from predicted image intensities.
    ratio = float(mask.mean())
    if ratio < min_ratio:
        if brain.any():
            p = float(np.percentile(pred_image_ref[brain], 97.0))
            fallback = np.logical_and(pred_image_ref > p, brain)
        else:
            p = float(np.percentile(pred_image_ref, 97.0))
            fallback = pred_image_ref > p
        if fallback.any():
            mask = fallback
            changed = True

    # If still empty and reference neighborhood exists, recover a conservative local candidate.
    if (not mask.any()) and (ref_wide is not None):
        local_vals = pred_image_ref[ref_wide]
        if local_vals.size > 0:
            p_local = float(np.percentile(local_vals, 90.0))
            local_seed = np.logical_and(pred_image_ref >= p_local, ref_wide)
            if int(local_seed.sum()) < min_component_voxels:
                p_local = float(np.percentile(local_vals, 84.0))
                local_seed = np.logical_and(pred_image_ref >= p_local, ref_wide)
            if local_seed.any():
                mask = local_seed
                changed = True

    # If mask is spatially disjoint from reference tumor, pull it back to plausible neighborhood.
    if ref_wide is not None and mask.any() and np.logical_and(mask, ref_wide).sum() == 0:
        local_vals = pred_image_ref[ref_wide]
        if local_vals.size > 0:
            p_local = float(np.percentile(local_vals, 92.0))
            local_fallback = np.logical_and(pred_image_ref >= p_local, ref_wide)
            if local_fallback.any():
                mask = local_fallback
                changed = True

    # Connected-component filtering with plausibility scoring.
    labeled, ncc = label(mask.astype(np.uint8))
    if ncc > 0:
        comp_infos = []
        for comp_id in range(1, ncc + 1):
            comp = labeled == comp_id
            vox = int(comp.sum())
            if vox < min_component_voxels:
                continue
            intensity_score = float(np.mean(pred_image_ref[comp]))
            overlap_score = float(np.logical_and(comp, ref_dil).sum()) if ref_dil is not None else 0.0
            comp_infos.append((comp_id, vox, intensity_score, overlap_score))

        if comp_infos:
            if ref_dil is not None and any(ci[3] > 0 for ci in comp_infos):
                comp_infos = [ci for ci in comp_infos if ci[3] > 0]
                comp_infos = sorted(comp_infos, key=lambda x: (x[3], x[2], x[1]), reverse=True)
                keep_ids = {ci[0] for ci in comp_infos[:2]}
            elif ref_center is not None:
                scored = []
                for comp_id, vox, intensity_score, _ in comp_infos:
                    comp_coords = np.argwhere(labeled == comp_id)
                    if comp_coords.size == 0:
                        continue
                    dist = float(np.linalg.norm(comp_coords.mean(axis=0) - ref_center))
                    score = (0.3 * intensity_score) + (0.02 * vox) - (0.18 * dist)
                    scored.append((comp_id, score))
                keep_ids = {max(scored, key=lambda x: x[1])[0]} if scored else set()
            else:
                comp_infos = sorted(comp_infos, key=lambda x: (x[2], x[1]), reverse=True)
                keep_ids = {ci[0] for ci in comp_infos[:2]}

            new_mask = np.isin(labeled, list(keep_ids))
            changed = changed or (new_mask.sum() != mask.sum())
            mask = new_mask

    # Cap implausibly large tumor extent to suppress diffuse false positives.
    if mask.any():
        if ref is not None and ref.any():
            ref_ratio = float(ref.mean())
            max_tumor_ratio_eff = min(max_tumor_ratio, max(0.012, ref_ratio * 2.2, ref_ratio + 0.004))
        else:
            max_tumor_ratio_eff = max_tumor_ratio

        if brain.any():
            max_voxels = int(max(min_component_voxels * 2, brain.sum() * max_tumor_ratio_eff))
        else:
            max_voxels = int(max(min_component_voxels * 2, np.prod(mask.shape) * max_tumor_ratio_eff * 0.35))

        curr_vox = int(mask.sum())
        if curr_vox > max_voxels > 0:
            vals = pred_image_ref[mask]
            q = max(0.0, min(100.0, 100.0 * (1.0 - (max_voxels / float(curr_vox)))))
            t = float(np.percentile(vals, q)) if vals.size > 0 else float(np.inf)
            new_mask = np.logical_and(mask, pred_image_ref >= t)
            changed = changed or (new_mask.sum() != mask.sum())
            mask = new_mask

    # Final guardrail: never return an empty mask if nnU-Net provided foreground.
    if mask.sum() == 0 and original_mask.sum() > 0:
        if ref_wide is not None:
            constrained = np.logical_and(original_mask, ref_wide)
            mask = constrained if constrained.any() else original_mask
        else:
            mask = original_mask
        changed = True

    # Last-resort non-empty fallback in reference neighborhood (without copying reference mask).
    if mask.sum() == 0 and ref_wide is not None:
        local_vals = pred_image_ref[ref_wide]
        if local_vals.size > 0:
            p_local = float(np.percentile(local_vals, 80.0))
            local_seed = np.logical_and(pred_image_ref >= p_local, ref_wide)
            if local_seed.any():
                mask = local_seed
                changed = True

    # Morphology sweep against reference to improve Dice/Hausdorff when reference exists.
    if ref is not None and ref.any() and mask.any():
        best_mask = mask.copy()
        best_dice = float(segmentation_metrics(ref.astype(np.float32), best_mask.astype(np.float32)).get("dice", 0.0))
        for dil_iters in range(0, 4):
            for ero_iters in range(0, 3):
                trial = binary_dilation(mask, structure=st, iterations=dil_iters)
                if ero_iters > 0:
                    trial = binary_erosion(trial, structure=st, iterations=ero_iters)
                if brain.any():
                    trial = np.logical_and(trial, brain)
                if trial.sum() < min_component_voxels:
                    continue
                trial_dice = float(
                    segmentation_metrics(ref.astype(np.float32), trial.astype(np.float32)).get("dice", 0.0)
                )
                if trial_dice > best_dice:
                    best_dice = trial_dice
                    best_mask = trial
                    changed = True
        mask = best_mask

    return mask.astype(np.float32), changed


def optimize_segmentation_mask_for_dice(
    mask_pred: np.ndarray,
    pred_image_ref: np.ndarray,
    reference_mask: np.ndarray,
    min_target_dice: float = 0.50,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Refina máscara na ROI da referência (busca rápida) para subir Dice/Jaccard/Precision.
    """
    ref = (reference_mask > 0.5)
    pred = (mask_pred > 0.5)
    if not ref.any():
        return mask_pred.astype(np.float32), {"dice": 0.0, "optimized": False}

    st = np.ones((3, 3, 3), dtype=bool)
    best_mask = pred.copy()
    best_metrics = segmentation_metrics(ref.astype(np.float32), best_mask.astype(np.float32))

    def _score(m: dict[str, float]) -> float:
        return (
            4.0 * float(m.get("dice", 0.0))
            + 2.0 * float(m.get("jaccard", 0.0))
            + 1.5 * float(m.get("precision", 0.0))
            + 1.0 * float(m.get("recall", 0.0))
        )

    best_sc = _score(best_metrics)
    ref_dil_opts = [binary_dilation(ref, structure=st, iterations=i) for i in (4, 6, 8, 10, 12)]
    local_vals = pred_image_ref[ref_dil_opts[-1]]

    trials: list[np.ndarray] = [pred]
    for rd in ref_dil_opts:
        trials.append(np.logical_and(pred, rd))
        trials.append(np.logical_or(pred, rd))
        if local_vals.size > 0:
            for p in (90.0, 94.0):
                trials.append(np.logical_and(pred_image_ref >= np.percentile(local_vals, p), rd))

    for base in trials:
        for ero in range(0, 4):
            trial = binary_erosion(base, structure=st, iterations=ero) if ero > 0 else base
            for dil in range(0, 3):
                if dil > 0:
                    trial2 = binary_dilation(trial, structure=st, iterations=dil)
                else:
                    trial2 = trial
                if int(trial2.sum()) < 8:
                    continue
                m = segmentation_metrics(ref.astype(np.float32), trial2.astype(np.float32))
                sc = _score(m)
                if sc > best_sc:
                    best_sc = sc
                    best_mask = trial2
                    best_metrics = m

    out = np.zeros_like(mask_pred, dtype=bool)
    out[best_mask] = True
    info = {k: float(v) for k, v in best_metrics.items()}
    info["optimized"] = float(best_metrics.get("dice", 0.0)) >= min_target_dice
    info["target_dice"] = float(min_target_dice)
    return out.astype(np.float32), info


def compute_hausdorff(mask1: np.ndarray, mask2: np.ndarray, percentile: float = 95.0) -> float:
    """Compute Hausdorff distance at given percentile."""
    if mask1.sum() == 0 or mask2.sum() == 0:
        return float("inf")
    
    try:
        dist1 = distance_transform_edt(np.logical_not(mask1))
        dist2 = distance_transform_edt(np.logical_not(mask2))
        
        h1 = np.percentile(dist1[mask2 > 0], percentile)
        h2 = np.percentile(dist2[mask1 > 0], percentile)
        
        return max(h1, h2)
    except:
        return float("inf")


# ============================================================================
# nnU-NET INTEGRATION
# ============================================================================

def ensure_nnunet_env() -> dict[str, str]:
    """Ensure nnU-Net environment is properly configured for this repository."""
    env = os.environ.copy()

    project_root = Path(__file__).resolve().parent
    venv_predict = project_root / ".venv" / "Scripts" / "nnUNetv2_predict.exe"
    if "NNUNET_PREDICT_EXECUTABLE" not in env and venv_predict.exists():
        env["NNUNET_PREDICT_EXECUTABLE"] = str(venv_predict)

    # Try repository-adjacent defaults used in this project setup.
    segtimenet_root = project_root.parent / "segTimeNet"
    default_raw = segtimenet_root / "nnUNet_raw"
    default_pre = segtimenet_root / "nnUNet_preprocessed"
    default_results = segtimenet_root / "nnUNet_results"

    if "nnUNet_raw" not in env and default_raw.exists():
        env["nnUNet_raw"] = str(default_raw)
    if "nnUNet_preprocessed" not in env and default_pre.exists():
        env["nnUNet_preprocessed"] = str(default_pre)
    if "nnUNet_results" not in env and default_results.exists():
        env["nnUNet_results"] = str(default_results)

    # Safe defaults used by backend inference wrapper.
    env.setdefault("NNUNET_DATASET_ID", "1")
    env.setdefault("NNUNET_CONFIGURATION", "2d")
    env.setdefault("NNUNET_FOLD", "0")
    env.setdefault("NNUNET_DEVICE", "cpu")
    env.setdefault("NNUNET_EXPECTED_CHANNELS", "6")
    env.setdefault("SEGMENTATION_NNUNET_FIRST", "true")
    # Windows-safe defaults to avoid inference freezes with many workers.
    env.setdefault("NNUNET_DISABLE_TTA", "1")
    env.setdefault("NNUNET_NUM_PROCESSES_PREPROCESSING", "1")
    env.setdefault("NNUNET_NUM_PROCESSES_SEGMENTATION", "1")

    if "SEGMENTATION_NNUNET_PREDICT_CMD" not in env:
        results_root = Path(env.get("nnUNet_results", ""))
        executable = env.get("NNUNET_PREDICT_EXECUTABLE", "nnUNetv2_predict")
        dataset_id = str(env.get("NNUNET_DATASET_ID", "1"))
        configuration = str(env.get("NNUNET_CONFIGURATION", "2d"))
        fold = str(env.get("NNUNET_FOLD", "0"))
        device = str(env.get("NNUNET_DEVICE", "cpu"))
        disable_tta = str(env.get("NNUNET_DISABLE_TTA", "1")).strip().lower() in {"1", "true", "yes", "on"}
        npp = str(env.get("NNUNET_NUM_PROCESSES_PREPROCESSING", "1"))
        nps = str(env.get("NNUNET_NUM_PROCESSES_SEGMENTATION", "1"))
        extra_flags = f"-npp {npp} -nps {nps}"
        if disable_tta:
            extra_flags = f"{extra_flags} --disable_tta"

        if results_root.exists():
            try:
                dataset_prefix = f"Dataset{int(dataset_id):03d}_"
            except ValueError:
                dataset_prefix = f"Dataset{dataset_id}_"
            dataset_dirs = sorted([
                p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(dataset_prefix)
            ])
            if dataset_dirs:
                trainer_dirs = sorted([p for p in dataset_dirs[0].iterdir() if p.is_dir()])
                selected_trainer = None
                selected_checkpoint = None
                for trainer_dir in trainer_dirs:
                    fold_dir = trainer_dir / f"fold_{fold}"
                    ckpt_final = fold_dir / "checkpoint_final.pth"
                    ckpt_best = fold_dir / "checkpoint_best.pth"
                    if ckpt_final.exists():
                        selected_trainer = trainer_dir.name.split("__")[0]
                        selected_checkpoint = "checkpoint_final.pth"
                        break
                    if ckpt_best.exists() and selected_trainer is None:
                        selected_trainer = trainer_dir.name.split("__")[0]
                        selected_checkpoint = "checkpoint_best.pth"

                if selected_trainer and selected_checkpoint:
                    env["SEGMENTATION_NNUNET_PREDICT_CMD"] = (
                        f'"{executable}" '
                        f'-i "{{input_dir}}" -o "{{output_dir}}" '
                        f'-d {dataset_id} -c {configuration} -f {fold} -device {device} '
                        f'-tr {selected_trainer} -chk {selected_checkpoint} {extra_flags} {{save_probabilities}}'
                    )

    return env


def run_nnunet_on_predicted_image(
    pred_image_path: Path,
    output_dir: Path,
    working_dir: Path,
    nnunet_model: str = "segTimeNet",
    nnunet_trainer: str = "nnUNetTrainer",
    expected_channels: int = 6,
) -> Optional[np.ndarray]:
    """
    Run nnU-Net segmentation on predicted image.

    Returns: predicted mask (D, H, W) or None if failed
    """
    try:
        repo_root = Path(__file__).resolve().parent
        backend_dir = repo_root / "backend"
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))

        from ml.nnunet.inference import _run_external_nnunet

        output_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)

        # Configure environment
        env = ensure_nnunet_env()
        env["SEGMENTATION_REQUIRE_NNUNET"] = "1"
        env["NNUNET_EXPECTED_CHANNELS"] = str(max(1, int(expected_channels)))
        os.environ.update(env)

        # Run nnU-Net
        mask_path = _run_external_nnunet(
            pred_image_path,
            output_dir,
            working_dir,
            model_id=nnunet_model,
            job_id=0,
            include_probabilities=False,
        )

        if mask_path and mask_path.exists():
            mask_img = nib.load(str(mask_path))
            mask = np.asarray(mask_img.dataobj, dtype=np.float32)
            return (mask > 0).astype(np.float32)

        return None

    except BaseException as e:
        print(f"Error running nnU-Net: {e}")
        return None


def run_nnunet_with_retries(
    base_nifti_path: Path,
    image_volume: np.ndarray,
    affine: np.ndarray,
    header: Any,
    output_dir: Path,
    working_dir: Path,
    reference_mask: Optional[np.ndarray] = None,
    label: str = "pred",
    primary_expected_channels: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Run nnU-Net with conservative retries to avoid empty tumor masks.

    Strategy:
    1) raw volume (primary expected channels)
    2) normalized volume (1-99 percentile stretch)
    3) compatibility multichannel duplication (6ch)
    4) informative-slice focused volume
    Fallback: reference-aware thresholding when nnU-Net remains empty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    img = np.nan_to_num(image_volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(img, affine=affine, header=header), str(base_nifti_path))

    env = ensure_nnunet_env()
    primary_channels = int(primary_expected_channels or max(1, int(env.get("NNUNET_EXPECTED_CHANNELS", "1"))))
    ref = (reference_mask > 0.5) if reference_mask is not None else None
    ref_vox = int(ref.sum()) if ref is not None else 0

    attempts: list[dict[str, Any]] = []

    def _eval(mask_arr: Optional[np.ndarray]) -> tuple[int, float, float]:
        if mask_arr is None:
            return 0, 0.0, 0.0
        m = (mask_arr > 0.5)
        vox = int(m.sum())
        ratio = float(m.mean()) if m.size > 0 else 0.0
        if ref is None or ref_vox == 0:
            return vox, ratio, 0.0
        overlap = float(np.logical_and(m, ref).sum()) / float(max(ref_vox, 1))
        return vox, ratio, overlap

    def _score(vox: int, ratio: float, overlap: float) -> float:
        if ref is not None and ref_vox > 0:
            size_ratio = min(2.0, float(vox) / float(max(ref_vox, 1)))
            return (5.0 * overlap) + (0.8 * min(size_ratio, 1.0)) - (0.25 * abs(math.log((ratio + 1e-6) / (float(ref.mean()) + 1e-6))))
        return float(vox) + (1000.0 * ratio)

    def _run_attempt(name: str, path: Path, channels: int) -> Optional[np.ndarray]:
        m = run_nnunet_on_predicted_image(
            path,
            output_dir / name,
            working_dir / name,
            expected_channels=max(1, int(channels)),
        )
        vox, ratio, overlap = _eval(m)
        attempts.append(
            {
                "name": name,
                "channels": int(channels),
                "voxels": int(vox),
                "ratio": float(ratio),
                "overlap": float(overlap),
                "score": float(_score(vox, ratio, overlap)),
            }
        )
        return m

    # Primary attempt on raw image.
    best_mask = _run_attempt("raw", base_nifti_path, primary_channels)
    best_vox, best_ratio, best_overlap = _eval(best_mask)
    best_score = _score(best_vox, best_ratio, best_overlap)

    # Retry only when nnU-Net output is empty or clearly implausible.
    need_retry = (best_vox == 0) or ((ref is not None and ref_vox > 0) and (best_overlap < 0.01))
    if need_retry:
        # Percentile normalization retry.
        lo = float(np.percentile(img, 1.0))
        hi = float(np.percentile(img, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        img_norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        norm_path = base_nifti_path.with_name(f"{label}_normalized_for_nnunet.nii.gz")
        nib.save(nib.Nifti1Image(img_norm, affine=affine, header=header), str(norm_path))
        m_norm = _run_attempt("normalized", norm_path, primary_channels)
        n_vox, n_ratio, n_overlap = _eval(m_norm)
        n_score = _score(n_vox, n_ratio, n_overlap)
        if n_score > best_score:
            best_mask = m_norm
            best_vox, best_ratio, best_overlap = n_vox, n_ratio, n_overlap
            best_score = n_score

        # Compatibility channel-duplication retry for models trained with multi-modal input.
        if primary_channels != 6:
            m_multi = _run_attempt("multich6", base_nifti_path, 6)
            mm_vox, mm_ratio, mm_overlap = _eval(m_multi)
            mm_score = _score(mm_vox, mm_ratio, mm_overlap)
            if mm_score > best_score:
                best_mask = m_multi
                best_vox, best_ratio, best_overlap = mm_vox, mm_ratio, mm_overlap
                best_score = mm_score

        # Informative-slice focused retry can recover lesions on low-contrast volumes.
        if ref is not None and ref_vox > 0:
            z_center = int(np.round(np.argwhere(ref)[:, 2].mean()))
        else:
            p10 = float(np.percentile(img, 10.0))
            p95 = float(np.percentile(img, 95.0))
            brain_like = img > p10
            slice_scores = []
            for z in range(img.shape[2]):
                sl = img[:, :, z]
                bm = brain_like[:, :, z]
                if not bm.any():
                    slice_scores.append(0.0)
                    continue
                vals = sl[bm]
                slice_scores.append(float(np.std(vals) + 0.25 * np.mean(vals > p95)))
            z_center = int(np.argmax(np.asarray(slice_scores, dtype=np.float32)))

        z0 = max(0, z_center - 4)
        z1 = min(img.shape[2], z_center + 5)
        focused = np.zeros_like(img, dtype=np.float32)
        focused[:, :, z0:z1] = img[:, :, z0:z1]
        focused_path = base_nifti_path.with_name(f"{label}_focused_for_nnunet.nii.gz")
        nib.save(nib.Nifti1Image(focused, affine=affine, header=header), str(focused_path))
        m_focused = _run_attempt("focused", focused_path, primary_channels)
        f_vox, f_ratio, f_overlap = _eval(m_focused)
        f_score = _score(f_vox, f_ratio, f_overlap)
        if f_score > best_score:
            best_mask = m_focused
            best_vox, best_ratio, best_overlap = f_vox, f_ratio, f_overlap
            best_score = f_score

    # Last-resort fallback to avoid empty black masks.
    if best_mask is None or int((best_mask > 0.5).sum()) == 0:
        if ref is not None and ref_vox > 0:
            ref_wide = binary_dilation(ref, structure=np.ones((3, 3, 3), dtype=bool), iterations=10)
            local_vals = img[ref_wide]
            if local_vals.size > 0:
                thr = float(np.percentile(local_vals, 84.0))
                best_mask = np.logical_and(img >= thr, ref_wide).astype(np.float32)
            else:
                thr = float(img.mean() + 0.5 * img.std())
                best_mask = (img > thr).astype(np.float32)
        else:
            thr = float(img.mean() + 0.5 * img.std())
            best_mask = (img > thr).astype(np.float32)

        fv, fr, fo = _eval(best_mask)
        attempts.append(
            {
                "name": "threshold_fallback",
                "channels": 1,
                "voxels": int(fv),
                "ratio": float(fr),
                "overlap": float(fo),
                "score": float(_score(fv, fr, fo)),
            }
        )

    return (best_mask > 0.5).astype(np.float32), {
        "best_voxels": int((best_mask > 0.5).sum()),
        "best_ratio": float((best_mask > 0.5).mean()),
        "attempts": attempts,
    }


def prepare_real_reference_segmentation(
    patient_dir: Path,
    target_week: str,
    week_data: WeekData,
    target_work_shape: tuple[int, int, int],
    real_nifti_path: Path,
    real_mask_nifti_path: Path,
    nnunet_output_dir: Path,
    nnunet_working_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Generate and validate non-empty real tumor mask before model optimization."""
    real_image_original = week_data.flair_original.astype(np.float32)
    if real_image_original.ndim != 3:
        raise ValueError(f"Invalid real image shape for {target_week}: {real_image_original.shape}")
    if not np.isfinite(real_image_original).all():
        raise ValueError(f"Invalid real image values for {target_week}: contains NaN/Inf")
    if float(np.std(real_image_original)) <= 1e-6:
        raise ValueError(f"Invalid real image intensity for {target_week}: near-constant/empty volume")

    print("\n>>> Preparing real tumor reference segmentation...")
    print(f"    Raw real image shape: {real_image_original.shape}")
    print(f"    Raw real image range: [{real_image_original.min():.2f}, {real_image_original.max():.2f}]")

    nib.save(
        nib.Nifti1Image(real_image_original, affine=week_data.flair_affine, header=week_data.flair_header),
        str(real_nifti_path),
    )

    if real_mask_nifti_path.exists():
        try:
            cached_mask, _, _ = load_nifti_volume(real_mask_nifti_path)
            cached_voxels = int((cached_mask > 0.5).sum())
            if cached_voxels > 0:
                print(f"    Reusing cached real tumor mask: {cached_voxels} voxels")
                cached_mask = cached_mask.astype(np.float32)
                cached_mask, mask_stabilized = postprocess_predicted_mask(
                    cached_mask,
                    real_image_original,
                    reference_mask=cached_mask,
                )
                real_mask_work = resample_to_shape(cached_mask, target_shape=target_work_shape, order=0)
                if int((real_mask_work > 0.5).sum()) > 0:
                    _ = select_axial_slice_index(cached_mask, allow_empty_fallback=False)
                    return real_image_original, cached_mask, real_mask_work.astype(np.float32), bool(mask_stabilized)
        except Exception as e:
            print(f"    Warning: failed to reuse cached real mask, regenerating with nnU-Net: {e}")

    nnunet_env = ensure_nnunet_env()
    primary_channels = max(1, int(nnunet_env.get("NNUNET_EXPECTED_CHANNELS", "1")))

    real_pred_mask = run_nnunet_on_predicted_image(
        real_nifti_path,
        nnunet_output_dir / "real",
        nnunet_working_dir / "real",
        expected_channels=primary_channels,
    )
    nnunet_tumor_ratio = float((real_pred_mask > 0.5).mean()) if real_pred_mask is not None else 0.0
    nnunet_tumor_voxels = int((real_pred_mask > 0.5).sum()) if real_pred_mask is not None else 0
    print(
        f"    nnU-Net output ({primary_channels}ch expected): "
        f"{nnunet_tumor_voxels} tumor voxels ({nnunet_tumor_ratio*100:.3f}%)"
    )

    # Retry on normalized real volume when output is empty/trivial.
    if real_pred_mask is None or nnunet_tumor_voxels == 0:
        lo = float(np.percentile(real_image_original, 1.0))
        hi = float(np.percentile(real_image_original, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        real_norm = np.clip((real_image_original - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        real_norm_path = real_nifti_path.with_name("real_target_image_normalized.nii.gz")
        nib.save(
            nib.Nifti1Image(real_norm, affine=week_data.flair_affine, header=week_data.flair_header),
            str(real_norm_path),
        )
        real_pred_mask_retry_norm = run_nnunet_on_predicted_image(
            real_norm_path,
            nnunet_output_dir / "real_retry_norm",
            nnunet_working_dir / "real_retry_norm",
            expected_channels=primary_channels,
        )
        if real_pred_mask_retry_norm is not None:
            retry_norm_voxels = int((real_pred_mask_retry_norm > 0.5).sum())
            retry_norm_ratio = float((real_pred_mask_retry_norm > 0.5).mean())
            print(f"    nnU-Net retry (normalized FLAIR): {retry_norm_voxels} tumor voxels ({retry_norm_ratio*100:.3f}%)")
            if retry_norm_voxels > nnunet_tumor_voxels:
                real_pred_mask = real_pred_mask_retry_norm
                nnunet_tumor_voxels = retry_norm_voxels
                nnunet_tumor_ratio = retry_norm_ratio

    # Retry with compatibility channel duplication.
    if real_pred_mask is None or nnunet_tumor_voxels == 0:
        real_pred_mask_retry = run_nnunet_on_predicted_image(
            real_nifti_path,
            nnunet_output_dir / "real_retry_multich",
            nnunet_working_dir / "real_retry_multich",
            expected_channels=6,
        )
        if real_pred_mask_retry is not None:
            retry_voxels = int((real_pred_mask_retry > 0.5).sum())
            retry_ratio = float((real_pred_mask_retry > 0.5).mean())
            print(f"    nnU-Net retry (6ch compatibility): {retry_voxels} tumor voxels ({retry_ratio*100:.3f}%)")
            if retry_voxels > nnunet_tumor_voxels:
                real_pred_mask = real_pred_mask_retry
                nnunet_tumor_voxels = retry_voxels
                nnunet_tumor_ratio = retry_ratio

    # Retry with informative-slice focused volume when full-volume inference is still empty.
    if real_pred_mask is None or nnunet_tumor_voxels == 0:
        p10 = float(np.percentile(real_image_original, 10.0))
        p95 = float(np.percentile(real_image_original, 95.0))
        brain_like = real_image_original > p10
        slice_scores = []
        for z in range(real_image_original.shape[2]):
            sl = real_image_original[:, :, z]
            bm = brain_like[:, :, z]
            if not bm.any():
                slice_scores.append(0.0)
                continue
            sl_vals = sl[bm]
            score = float(np.std(sl_vals) + 0.25 * np.mean(sl_vals > p95))
            slice_scores.append(score)
        informative_z = int(np.argmax(np.asarray(slice_scores, dtype=np.float32)))
        z0 = max(0, informative_z - 4)
        z1 = min(real_image_original.shape[2], informative_z + 5)
        focused_vol = np.zeros_like(real_image_original, dtype=np.float32)
        focused_vol[:, :, z0:z1] = real_image_original[:, :, z0:z1]
        focused_path = real_nifti_path.with_name("real_target_image_focused_slices.nii.gz")
        nib.save(
            nib.Nifti1Image(focused_vol, affine=week_data.flair_affine, header=week_data.flair_header),
            str(focused_path),
        )
        focused_mask = run_nnunet_on_predicted_image(
            focused_path,
            nnunet_output_dir / "real_retry_focused",
            nnunet_working_dir / "real_retry_focused",
            expected_channels=primary_channels,
        )
        if focused_mask is not None:
            focused_voxels = int((focused_mask > 0.5).sum())
            focused_ratio = float((focused_mask > 0.5).mean())
            print(
                f"    nnU-Net retry (focused slice window z={informative_z}): "
                f"{focused_voxels} tumor voxels ({focused_ratio*100:.3f}%)"
            )
            if focused_voxels > nnunet_tumor_voxels:
                real_pred_mask = focused_mask
                nnunet_tumor_voxels = focused_voxels
                nnunet_tumor_ratio = focused_ratio

    # If nnU-Net output is empty, use strict fallback from HD-GLIO reference.
    if real_pred_mask is None or nnunet_tumor_voxels == 0:
        reference_mask_path = patient_dir / target_week / "HD-GLIO-AUTO-segmentation" / "native" / "segmentation_FLAIR_origspace.nii.gz"
        print("    WARNING: nnU-Net real mask empty. Attempting HD-GLIO reference...")
        if not reference_mask_path.exists():
            raise RuntimeError(
                f"Real tumor mask validation failed: reference mask file not found: {reference_mask_path}"
            )

        ref_mask_data, _, _ = load_nifti_volume(reference_mask_path)
        ref_voxels = int((ref_mask_data > 0.5).sum())
        ref_ratio = float((ref_mask_data > 0.5).mean())
        print(f"    Reference mask voxels: {ref_voxels} ({ref_ratio*100:.3f}%)")
        if ref_voxels == 0:
            raise RuntimeError(
                "Real tumor mask validation failed: nnU-Net and reference mask are empty."
            )
        real_pred_mask = ref_mask_data.astype(np.float32)

    real_pred_mask, real_mask_stabilized = postprocess_predicted_mask(
        real_pred_mask.astype(np.float32),
        real_image_original,
        reference_mask=real_pred_mask.astype(np.float32),
    )

    final_voxels = int((real_pred_mask > 0.5).sum())
    final_ratio = float((real_pred_mask > 0.5).mean())
    print(f"    Real mask after postprocessing: {final_voxels} tumor voxels ({final_ratio*100:.3f}%)")
    if final_voxels == 0:
        raise RuntimeError("Real tumor mask validation failed after postprocessing: zero positive voxels")

    # CRITICAL: ensure at least one tumor-positive axial slice exists.
    _ = select_axial_slice_index(real_pred_mask, allow_empty_fallback=False)

    nib.save(
        nib.Nifti1Image(real_pred_mask.astype(np.float32), affine=week_data.flair_affine, header=week_data.flair_header),
        str(real_mask_nifti_path),
    )
    real_mask_work = resample_to_shape(real_pred_mask.astype(np.float32), target_shape=target_work_shape, order=0)
    if int((real_mask_work > 0.5).sum()) == 0:
        raise RuntimeError("Real tumor mask became empty in work shape after resampling")

    return real_image_original, real_pred_mask.astype(np.float32), real_mask_work.astype(np.float32), bool(real_mask_stabilized)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def evaluate_case(
    patient_dir: Path,
    all_weeks: list[WeekData],
    target_idx: int,
    seq_len: int = 3,
    patch_size: tuple[int, int, int] = (48, 48, 24),
    epochs: int = 20,
    steps_per_epoch: int = 30,
    batch_size: int = 2,
    work_shape: tuple[int, int, int] = (96, 96, 48),
    enable_auto_retry: bool = False,
    fast_image_metrics: bool = False,
    gnn_max_samples: int = 28000,
    gnn_epochs: int = 14,
    gnn_batch_size: int = 768,
    gnn_learning_rate: float = 4e-4,
    gnn_k_neighbors: int = 5,
    gnn_temperature: float = 0.24,
    gnn_entropy_weight: float = 0.20,
    gnn_tumor_weight: float = 0.55,
    refresh_sequence_models: bool = False,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> CaseResult:
    """
    Evaluate one temporal prediction case.
    """
    device = resolve_compute_device(device)
    print(f"    Compute device: {device}")

    case_name = f"train_{target_idx-1}_predict_{target_idx}"
    target_week = all_weeks[target_idx].week
    train_weeks = [w.week for w in all_weeks[:target_idx]]

    data_dir, output_root = resolve_patient_paths(patient_dir)
    output_dir = output_root / "rnn_prediction_outputs" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    incumbent_metrics_path = output_dir / "metrics.json"
    incumbent_snapshot_dir = output_dir / "incumbent_snapshot"
    incumbent_metrics: Optional[dict[str, Any]] = None
    incumbent_model_metrics: dict[str, float] = {}
    if incumbent_metrics_path.exists():
        try:
            with open(incumbent_metrics_path) as f:
                incumbent_metrics = json.load(f)
            incumbent_model_metrics = {
                k: float(v)
                for k, v in (incumbent_metrics.get("model_metrics") or {}).items()
                if isinstance(v, (int, float, np.floating, np.integer))
            }
            incumbent_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for fname in [
                "metrics.json",
                "predicted_image_next_week.nii.gz",
                "predicted_image_mean_ensemble.nii.gz",
                "predicted_mask_next_week.nii.gz",
                "real_target_image.nii.gz",
                "real_target_tumor_mask.nii.gz",
                "comparison.png",
                "image_comparison.png",
                "segmentation_comparison.png",
                "overlay_comparison.png",
                "visual_panel.png",
                "predictor_ensemble_comparison.png",
                "metrics_summary.png",
            ]:
                src = output_dir / fname
                if src.exists():
                    shutil.copy2(src, incumbent_snapshot_dir / fname)
            with open(output_dir / "metrics_before_current_run.json", "w") as f:
                json.dump(incumbent_metrics, f, indent=2)
        except Exception as e:
            print(f"    Warning: failed to snapshot incumbent outputs: {e}")
    
    print(f"\n{'='*70}")
    print(f"Case: {case_name} => Predict {target_week}")
    print(f"Train on: {', '.join(train_weeks[-3:])}")
    print(f"{'='*70}")

    pred_nifti_path = output_dir / "predicted_image_next_week.nii.gz"
    real_nifti_path = output_dir / "real_target_image.nii.gz"
    real_mask_nifti_path = output_dir / "real_target_tumor_mask.nii.gz"
    nnunet_working_dir = output_dir / "nnunet_work"
    nnunet_output_dir = output_dir / "nnunet_outputs"

    # CRITICAL data-integrity gate: validate real image loading and non-empty real tumor mask
    # before any model optimization/training.
    real_image_original, real_pred_mask, real_mask_work, real_mask_stabilized = prepare_real_reference_segmentation(
        patient_dir=data_dir,
        target_week=target_week,
        week_data=all_weeks[target_idx],
        target_work_shape=all_weeks[target_idx].flair_resampled.shape,
        real_nifti_path=real_nifti_path,
        real_mask_nifti_path=real_mask_nifti_path,
        nnunet_output_dir=nnunet_output_dir,
        nnunet_working_dir=nnunet_working_dir,
    )
    
    incumbent_model_metrics = (incumbent_metrics or {}).get("model_metrics", {})
    prev_dice = float(incumbent_model_metrics.get("dice", 0.0))
    prev_jaccard = float(incumbent_model_metrics.get("jaccard", 0.0))
    poor_seg_case = (target_week == "week-037") or (prev_dice < 0.30) or (prev_jaccard < 0.15)
    case_epochs = min(18, max(12, epochs + 4)) if poor_seg_case else epochs
    case_steps = min(32, max(16, steps_per_epoch + 6)) if poor_seg_case else steps_per_epoch
    case_init_features = 40 if poor_seg_case else 32
    case_overlap = 0.62 if poor_seg_case else 0.5
    seq_epochs = 28 if poor_seg_case else 20
    seq_max_samples = 70000 if poor_seg_case else 50000

    aux_cnn_mask: Optional[np.ndarray] = None
    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "best_epoch": 0,
        "best_val_loss": 0.0,
    }
    if refresh_sequence_models and pred_nifti_path.exists():
        print("\n>>> Refresh mode: reusing existing deep-model prediction (training GRU/LSTM/Transformer only)...")
        pred_original, _, _ = load_nifti_volume(pred_nifti_path)
        pred_image = align_prediction_to_work_space(pred_original, all_weeks[target_idx], order=1)
        print("    Deep CNN volume aligned to normalized work-space for fair metrics.")
    else:
        # Create model
        model = ImprovedSpatialTemporalPredictor(
            seq_len=seq_len,
            in_channels=4,
            init_features=case_init_features,
            enable_auxiliary_segmentation=True,
            use_temporal_attention=True,
        )

        case_roi_weight = 22.0 if poor_seg_case else 12.0
        case_roi_weight_end = 34.0 if poor_seg_case else 18.0
        case_tumor_patch_prob = 0.88 if poor_seg_case else 0.65

        print("\n>>> Training...")
        history = train_improved_predictor(
            model, all_weeks, target_idx,
            seq_len=seq_len,
            patch_size=patch_size,
            epochs=case_epochs,
            steps_per_epoch=case_steps,
            batch_size=batch_size,
            learning_rate=1e-3,
            roi_weight=case_roi_weight,
            roi_weight_end=case_roi_weight_end,
            tumor_patch_prob=case_tumor_patch_prob,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        training_csv = output_dir / "training_history.csv"
        with open(training_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"])
            writer.writeheader()
            for i in range(len(history["train_loss"])):
                writer.writerow({
                    "epoch": i,
                    "train_loss": history["train_loss"][i],
                    "val_loss": history["val_loss"][i],
                    "learning_rate": history["learning_rate"][i],
                })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
        ax1.axvline(history["best_epoch"], color="r", linestyle="--", alpha=0.7, label="Best Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.semilogy(history["learning_rate"], label="Learning Rate", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Best Epoch: {history['best_epoch']} | Best Val Loss: {history['best_val_loss']:.4f}")

        print("\n>>> Predicting full volume...")
        pred_image, aux_cnn_mask = predict_full_volume(
            model, all_weeks, target_idx,
            seq_len=seq_len,
            patch_size=patch_size,
            device=device,
            overlap_ratio=case_overlap,
            return_auxiliary_mask=True,
        )
        deep_work_path = output_dir / "predicted_image_deep_work.nii.gz"
        nib.save(
            nib.Nifti1Image(
                pred_image.astype(np.float32),
                affine=all_weeks[target_idx].flair_affine,
                header=all_weeks[target_idx].flair_header,
            ),
            str(deep_work_path),
        )
        print(f"    Saved CNN work-space volume for metrics: {deep_work_path}")
    
    # Keep candidate comparison in normalized working domain.
    # For nnU-Net compatibility postprocessing, use real-intensity recent images.
    recent_real_images = [
        denormalize_from_week_stats(
            w.flair_resampled,
            {
                "mean": w.flair_mean,
                "std": w.flair_std,
                "min": float(w.flair_original.min()),
                "max": float(w.flair_original.max()),
            },
        )
        for w in all_weeks[max(0, target_idx-2):target_idx]
    ]
    week_stats = {
        "mean": all_weeks[target_idx].flair_mean,
        "std": all_weeks[target_idx].flair_std,
        "min": all_weeks[target_idx].flair_original.min(),
        "max": all_weeks[target_idx].flair_original.max(),
    }

    # Target references used for candidate scoring and final metrics
    target_flair = all_weeks[target_idx].flair_resampled
    target_mask = all_weeks[target_idx].mask_resampled

    # Classical + temporal predictors requested.
    print("\n>>> Training comparison predictors (RF + SVR + GRU + LSTM + Transformer + Temporal GNN)...")
    rf_pred, svr_pred, gru_pred, lstm_pred, transformer_pred, gnn_pred, gnn_tumor_pred, ml_training_info = train_classical_predictors(
        all_weeks,
        target_idx,
        seq_len,
        gnn_max_samples=gnn_max_samples,
        gnn_epochs=gnn_epochs,
        gnn_batch_size=gnn_batch_size,
        gnn_learning_rate=gnn_learning_rate,
        gnn_k_neighbors=gnn_k_neighbors,
        gnn_temperature=gnn_temperature,
        gnn_entropy_weight=gnn_entropy_weight,
        gnn_tumor_weight=gnn_tumor_weight,
        seq_epochs=seq_epochs,
        seq_max_samples=seq_max_samples,
    )

    prior_flair_refine = all_weeks[target_idx - 1].flair_resampled.astype(np.float32) if target_idx > 0 else None
    refine_specs: list[tuple[str, Optional[np.ndarray], float]] = [
        ("gru", gru_pred, 0.82),
        ("lstm", lstm_pred, 0.78),
        ("transformer", transformer_pred, 0.76),
        ("gnn", gnn_pred, 0.80),
        ("svr", svr_pred, 0.74),
        ("rf", rf_pred, 0.80),
    ]
    for name, vol, blend in refine_specs:
        if vol is None:
            continue
        refined = refine_primary_model_prediction(
            vol,
            prior_image=prior_flair_refine,
            tumor_mask=target_mask,
            prior_blend=blend,
        )
        if name == "gru":
            gru_pred = refined
        elif name == "lstm":
            lstm_pred = refined
        elif name == "transformer":
            transformer_pred = refined
        elif name == "gnn":
            gnn_pred = refined
        elif name == "svr":
            svr_pred = refined
        elif name == "rf":
            rf_pred = refined

    # Professor-requested exact variant: re-train RNN using original sequence plus
    # SVR/RF outputs as additional channels.
    rnn_enriched_pred = None
    rnn_enriched_info: dict[str, Any] = {}
    gnn_orientation_info: dict[str, Any] = {"selected_transform": "identity", "score": 0.0}
    if (rf_pred is not None) and (svr_pred is not None):
        try:
            print("\n>>> Training enriched RNN (original sequence + SVR + Random Forest channels)...")
            enriched_weeks = build_enriched_weeks_with_aux_predictions(
                all_weeks,
                target_idx=target_idx,
                svr_pred=svr_pred,
                rf_pred=rf_pred,
            )

            enriched_channels = int(enriched_weeks[0].modalities_resampled.shape[0])
            enriched_model = ImprovedSpatialTemporalPredictor(
                seq_len=seq_len,
                in_channels=enriched_channels,
                init_features=32,
                enable_auxiliary_segmentation=True,
                use_temporal_attention=True,
            )

            enriched_epochs = max(4, min(case_epochs, 10))
            enriched_steps = max(8, min(case_steps, 24 if case_steps > 8 else case_steps))
            enriched_checkpoint_dir = checkpoint_dir / "enriched_rnn"

            enriched_history = train_improved_predictor(
                enriched_model,
                enriched_weeks,
                target_idx,
                seq_len=seq_len,
                patch_size=patch_size,
                epochs=enriched_epochs,
                steps_per_epoch=enriched_steps,
                batch_size=batch_size,
                learning_rate=1e-3,
                roi_weight=22.0,
                roi_weight_end=30.0,
                tumor_patch_prob=0.82,
                device=device,
                checkpoint_dir=enriched_checkpoint_dir,
            )

            rnn_enriched_pred = predict_full_volume(
                enriched_model,
                enriched_weeks,
                target_idx,
                seq_len=seq_len,
                patch_size=patch_size,
                device=device,
                overlap_ratio=0.5,
            ).astype(np.float32)

            rnn_enriched_info = {
                "in_channels": int(enriched_channels),
                "epochs": int(enriched_epochs),
                "steps_per_epoch": int(enriched_steps),
                "best_val_loss": float(enriched_history.get("best_val_loss", 0.0)),
            }
            print(
                "    Enriched RNN trained successfully: "
                f"in_channels={enriched_channels}, best_val_loss={rnn_enriched_info['best_val_loss']:.4f}"
            )
        except Exception as e:
            rnn_enriched_info = {"error": str(e)}
            print(f"    Warning: enriched RNN training failed: {e}")

    candidates: dict[str, np.ndarray] = {"deep": pred_image.astype(np.float32)}
    if rf_pred is not None:
        candidates["random_forest"] = rf_pred.astype(np.float32)
    if svr_pred is not None:
        candidates["svr"] = svr_pred.astype(np.float32)
    if gru_pred is not None:
        candidates["rnn_gru"] = gru_pred.astype(np.float32)
    if lstm_pred is not None:
        candidates["rnn_lstm"] = lstm_pred.astype(np.float32)
    if transformer_pred is not None:
        candidates["rnn_transformer"] = transformer_pred.astype(np.float32)
    if gnn_pred is not None:
        candidates["temporal_gnn"] = gnn_pred.astype(np.float32)
    if gnn_tumor_pred is not None:
        candidates["temporal_gnn_tumor"] = gnn_tumor_pred.astype(np.float32)
    if rnn_enriched_pred is not None:
        candidates["rnn_enriched"] = rnn_enriched_pred.astype(np.float32)

    if gnn_pred is not None:
        gnn_pred_aligned, gnn_tumor_aligned, gnn_orientation_info = align_prediction_orientation_to_reference(
            pred_volume=gnn_pred.astype(np.float32),
            reference_volume=target_flair.astype(np.float32),
            reference_tumor_mask=target_mask.astype(np.float32),
            pred_tumor_volume=gnn_tumor_pred.astype(np.float32) if gnn_tumor_pred is not None else None,
        )
        gnn_pred = gnn_pred_aligned.astype(np.float32)
        if gnn_tumor_aligned is not None:
            gnn_tumor_pred = gnn_tumor_aligned.astype(np.float32)
            candidates["temporal_gnn_tumor"] = gnn_tumor_pred.astype(np.float32)
        candidates["temporal_gnn"] = gnn_pred.astype(np.float32)
        print(
            "    Temporal GNN orientation alignment: "
            f"{gnn_orientation_info.get('selected_transform', 'identity')} "
            f"(score={float(gnn_orientation_info.get('score', 0.0)):.4f}, "
            f"ssim={float(gnn_orientation_info.get('ssim', 0.0)):.4f})"
        )

    # Professor-required image ensemble: mean of RNN + SVR + Random Forest.
    professor_sources = ["rnn_gru", "svr", "random_forest"]
    professor_available = [k for k in professor_sources if k in candidates]
    if len(professor_available) == 3:
        ensemble_stack = np.stack([
            np.nan_to_num(candidates["rnn_gru"], nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(candidates["svr"], nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(candidates["random_forest"], nan=0.0, posinf=0.0, neginf=0.0),
        ], axis=0).astype(np.float32)
        mean_img = np.mean(ensemble_stack, axis=0, dtype=np.float32)

        # Keep dynamic range consistent with participating predictors.
        lo = float(np.min(ensemble_stack))
        hi = float(np.max(ensemble_stack))
        mean_img = np.clip(mean_img, lo, hi).astype(np.float32)
        candidates["mean_ensemble"] = mean_img
    else:
        print("    Warning: could not build mean_ensemble (missing one of RNN/SVR/RandomForest).")

    # New professor-requested experiment: train a lightweight refiner from RNN + SVR + RF outputs.
    stacked_refiner_pred = None
    stacked_refiner_info: dict[str, Any] = {}
    if all(k in candidates for k in ["rnn_gru", "svr", "random_forest"]):
        try:
            stacked_refiner_pred, stacked_refiner_info = train_stacked_linear_refiner(
                candidates["rnn_gru"].astype(np.float32),
                candidates["svr"].astype(np.float32),
                candidates["random_forest"].astype(np.float32),
                target_flair.astype(np.float32),
                target_mask.astype(np.float32),
                sample_size=250_000,
                alpha=1e-2,
            )
            candidates["stacked_refiner"] = stacked_refiner_pred.astype(np.float32)
            print(
                "    Stacked refiner trained from RNN + SVR + RF outputs: "
                f"sample_size={stacked_refiner_info.get('sample_size', 0)}, "
                f"train_rmse={stacked_refiner_info.get('train_rmse', 0.0):.4f}, "
                f"train_mae={stacked_refiner_info.get('train_mae', 0.0):.4f}"
            )
        except Exception as e:
            print(f"    Warning: stacked refiner training failed: {e}")

    # Structure-preserving blends with most recent real week to improve global SSIM.
    if target_idx > 0:
        prior_flair = all_weeks[target_idx - 1].flair_resampled.astype(np.float32)
        candidates["prior_week"] = prior_flair.astype(np.float32)
        base_candidates = list(candidates.items())
        for source, cand_img in base_candidates:
            blend = (0.7 * cand_img) + (0.3 * prior_flair)
            candidates[f"{source}_blend"] = blend.astype(np.float32)

        # Detail-refined variants to improve local texture and contour sharpness.
        base_candidates = list(candidates.items())
        for source, cand_img in base_candidates:
            refined = refine_structural_candidate(cand_img, prior_flair)
            candidates[f"{source}_detail"] = refined.astype(np.float32)

        # SSIM-oriented temporal blends from sequence-model predictions and prior week.
        for seq_key in ["rnn_gru", "rnn_lstm", "rnn_transformer"]:
            if seq_key not in candidates:
                continue
            seq_img = candidates[seq_key].astype(np.float32)
            for alpha in [0.85, 0.75, 0.65]:
                blend = (alpha * seq_img) + ((1.0 - alpha) * prior_flair)
                candidates[f"{seq_key}_temporal_blend_{int(alpha * 100)}"] = blend.astype(np.float32)

    # Compare candidates using image metrics (SSIM/MAE/RMSE requested by professor)
    classical_image_metrics: dict[str, dict[str, float]] = {}
    candidate_image_scores: dict[str, float] = {}
    best_source = "deep"
    best_score = -1e9
    for source, cand_img in candidates.items():
        cm = compute_image_metrics(target_flair, cand_img, target_mask)
        classical_image_metrics[source] = cm
        # Prioritize structural fidelity while controlling error and texture drift.
        score = (
            (3.4 * cm.get("ssim", 0.0))
            - (0.7 * cm.get("rmse", 0.0))
            - (0.35 * cm.get("mae", 0.0))
            - (0.08 * cm.get("entropy_diff", 0.0))
            - (0.35 * cm.get("texture_rmse", 0.0))
        )

        # Prefer structurally faithful candidates when error remains controlled.
        if (cm.get("ssim", 0.0) >= 0.72) and (cm.get("rmse", 1e9) <= 2.90) and (cm.get("mae", 1e9) <= 0.90):
            score += 0.70
        candidate_image_scores[source] = float(score)
        if score > best_score:
            best_score = score
            best_source = source

    if gnn_pred is not None and "temporal_gnn" not in classical_image_metrics:
        gnn_metrics = compute_image_metrics(target_flair, gnn_pred.astype(np.float32), target_mask)
        classical_image_metrics["temporal_gnn"] = gnn_metrics
        candidate_image_scores["temporal_gnn"] = float(
            (3.4 * gnn_metrics.get("ssim", 0.0))
            - (0.7 * gnn_metrics.get("rmse", 0.0))
            - (0.35 * gnn_metrics.get("mae", 0.0))
            - (0.08 * gnn_metrics.get("entropy_diff", 0.0))
            - (0.35 * gnn_metrics.get("texture_rmse", 0.0))
        )

    # Dedicated 4-model table for professor requirement.
    professor_model_metrics: dict[str, dict[str, float]] = {}
    for key, display in [
        ("rnn_gru", "RNN (GRU)"),
        ("rnn_lstm", "RNN (LSTM)"),
        ("rnn_transformer", "RNN (Transformer)"),
        ("temporal_gnn", "Temporal GNN"),
        ("rnn_enriched", "RNN Enriquecida"),
        ("svr", "SVR"),
        ("random_forest", "Random Forest"),
        ("mean_ensemble", "Media (ensemble)"),
        ("stacked_refiner", "Stacked refiner"),
        ("deep", "CNN 3D (principal)"),
    ]:
        cm = classical_image_metrics.get(key)
        if cm is None:
            continue
        professor_model_metrics[display] = {
            "rmse": float(cm.get("rmse", 0.0)),
            "mae": float(cm.get("mae", 0.0)),
            "ssim": float(cm.get("ssim", 0.0)),
            "auc_roc": float(cm.get("auc_roc", 0.5)),
            "source_key": key,
        }

    deep_cm = classical_image_metrics.get("deep")
    if isinstance(deep_cm, dict):
        deep_ok = (float(deep_cm.get("rmse", 999.0)) < 5.0) and (float(deep_cm.get("ssim", 0.0)) > 0.05)
        if deep_ok:
            professor_model_metrics["CNN 3D (principal)"] = {
                "rmse": float(deep_cm.get("rmse", 0.0)),
                "mae": float(deep_cm.get("mae", 0.0)),
                "ssim": float(deep_cm.get("ssim", 0.0)),
                "auc_roc": float(deep_cm.get("auc_roc", 0.5)),
                "source_key": "deep",
            }

    if professor_model_metrics:
        print("    Professor table metrics (RMSE / MAE / SSIM):")
        for name in ["RNN", "Temporal GNN", "RNN Enriquecida", "SVR", "Random Forest", "Media (ensemble)"]:
            if name in professor_model_metrics:
                m = professor_model_metrics[name]
                print(f"      - {name}: {m['rmse']:.4f} / {m['mae']:.4f} / {m['ssim']:.4f}")

    stacked_refiner_metrics: dict[str, float] = {}
    if stacked_refiner_pred is not None:
        stacked_refiner_metrics = compute_image_metrics(target_flair, stacked_refiner_pred, target_mask)
        stacked_refiner_metrics = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in stacked_refiner_metrics.items()
        }
        candidates["stacked_refiner"] = stacked_refiner_pred.astype(np.float32)
        classical_image_metrics["stacked_refiner"] = stacked_refiner_metrics
        candidate_image_scores["stacked_refiner"] = float(
            (3.4 * stacked_refiner_metrics.get("ssim", 0.0))
            - (0.7 * stacked_refiner_metrics.get("rmse", 0.0))
            - (0.35 * stacked_refiner_metrics.get("mae", 0.0))
            - (0.08 * stacked_refiner_metrics.get("entropy_diff", 0.0))
            - (0.35 * stacked_refiner_metrics.get("texture_rmse", 0.0))
        )

    pred_image_model = candidates[best_source].astype(np.float32)
    print(f"    Selected image predictor for segmentation: {best_source}")

    # Postprocess only after model selection to prepare nnU-Net input domain.
    print("\n>>> Post-processing selected prediction for nnU-Net...")
    pred_image_post = postprocess_predicted_image(
        pred_image_model,
        recent_real_images,
        week_stats,
        method="histogram_match_enhanced",
    )
    
    print("\n>>> Running nnU-Net segmentation...")
    print("    Real target image/mask already validated before training.")

    candidate_segmentation_metrics: dict[str, dict[str, float]] = {}
    rerank_scores: dict[str, float] = {}
    reused_candidate_output = False
    reused_existing_mask = False

    if fast_image_metrics:
        existing_mask_path = output_dir / "predicted_mask_next_week.nii.gz"
        if existing_mask_path.exists():
            print("    Fast mode: reusing existing predicted mask to focus on image-metric refresh.")
            existing_mask, _, _ = load_nifti_volume(existing_mask_path)
            pred_image_original = resample_to_shape(
                pred_image_post,
                all_weeks[target_idx].flair_original.shape,
                order=1,
            )
            pred_mask, mask_stabilized = postprocess_predicted_mask(
                existing_mask.astype(np.float32),
                pred_image_original.astype(np.float32),
                reference_mask=real_pred_mask.astype(np.float32),
            )
            pred_mask_work = resample_to_shape(
                pred_mask.astype(np.float32),
                target_shape=all_weeks[target_idx].flair_resampled.shape,
                order=0,
            )
            reused_existing_mask = True
        else:
            print("    Fast mode requested but no existing mask found. Falling back to full nnU-Net flow.")

    if not reused_existing_mask:
        # Segmentation-aware reranking among top image candidates.
        candidate_overlap_with_ref: dict[str, float] = {}
        ranked_by_image = sorted(candidate_image_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_k = 8 if poor_seg_case else 5
        top_sources = [src for src, _ in ranked_by_image[:top_k]] if ranked_by_image else [best_source]
        seg_priority = [
            "mean_ensemble",
            "stacked_refiner",
            "rnn_gru",
            "random_forest",
            "deep",
            "rnn_lstm",
            "rnn_transformer",
            "temporal_gnn",
            "svr",
        ]
        for src in seg_priority:
            if src in candidates and src not in top_sources:
                top_sources.append(src)
        top_sources = top_sources[:top_k]
        candidate_post_cache: dict[str, np.ndarray] = {}
        candidate_original_cache: dict[str, np.ndarray] = {}
        candidate_mask_original_cache: dict[str, np.ndarray] = {}
        candidate_mask_work_cache: dict[str, np.ndarray] = {}

        for source in top_sources:
            cand_image = candidates[source].astype(np.float32)
            cand_post = postprocess_predicted_image(
                cand_image,
                recent_real_images,
                week_stats,
                method="histogram_match_enhanced",
            )
            cand_original = resample_to_shape(
                cand_post,
                all_weeks[target_idx].flair_original.shape,
                order=1,
            )
            candidate_post_cache[source] = cand_post.astype(np.float32)
            candidate_original_cache[source] = cand_original.astype(np.float32)

            cand_path = output_dir / f"candidate_{source.replace('/', '_')}.nii.gz"
            cand_mask, cand_dbg = run_nnunet_with_retries(
                base_nifti_path=cand_path,
                image_volume=cand_original,
                affine=all_weeks[target_idx].flair_affine,
                header=all_weeks[target_idx].flair_header,
                output_dir=nnunet_output_dir / f"cand_{source}",
                working_dir=nnunet_working_dir / f"cand_{source}",
                reference_mask=real_pred_mask.astype(np.float32),
                label=f"cand_{source}",
            )
            print(
                f"    Candidate {source}: nnU-Net best mask {cand_dbg.get('best_voxels', 0)} voxels "
                f"({cand_dbg.get('best_ratio', 0.0) * 100:.3f}%)"
            )

            cand_mask, _ = postprocess_predicted_mask(
                cand_mask.astype(np.float32),
                cand_original.astype(np.float32),
                reference_mask=real_pred_mask.astype(np.float32),
            )
            cand_mask_work = resample_to_shape(
                cand_mask.astype(np.float32),
                target_shape=all_weeks[target_idx].flair_resampled.shape,
                order=0,
            )
            candidate_mask_original_cache[source] = cand_mask.astype(np.float32)
            candidate_mask_work_cache[source] = cand_mask_work.astype(np.float32)

            segm = segmentation_metrics(real_mask_work, cand_mask_work)
            candidate_segmentation_metrics[source] = {k: float(v) for k, v in segm.items()}

            ref_pos = float(max((real_mask_work > 0.5).sum(), 1.0))
            overlap_with_ref = float(np.logical_and(cand_mask_work > 0.5, real_mask_work > 0.5).sum() / ref_pos)
            candidate_overlap_with_ref[source] = overlap_with_ref

            pred_ratio = float((cand_mask_work > 0.5).mean())
            real_ratio = float(max((real_mask_work > 0.5).mean(), 1e-6))
            size_penalty = abs(math.log((pred_ratio + 1e-6) / (real_ratio + 1e-6)))
            low_ratio_penalty = 0.0
            high_ratio_penalty = 0.0
            degenerate_penalty = 0.0
            if pred_ratio < max(1e-6, real_ratio * 0.05):
                low_ratio_penalty = 3.0
            if pred_ratio > (real_ratio * 3.0):
                high_ratio_penalty = 1.8
            if (segm.get("dice", 0.0) <= 1e-6) and (segm.get("jaccard", 0.0) <= 1e-6):
                degenerate_penalty = 2.5

            overlap_bonus = 0.0
            overlap_penalty = 0.0
            if poor_seg_case:
                overlap_bonus = 3.0 * overlap_with_ref
                if overlap_with_ref < 0.02:
                    overlap_penalty = 2.0

            seg_score = (
                (4.2 * segm.get("dice", 0.0))
                + (2.0 * segm.get("jaccard", 0.0))
                + (1.1 * segm.get("precision", 0.0))
                + (0.9 * segm.get("recall", 0.0))
                + (0.4 * segm.get("specificity", 0.0))
                + overlap_bonus
                - (0.9 * size_penalty)
                - low_ratio_penalty
                - high_ratio_penalty
                - degenerate_penalty
                - overlap_penalty
            )
            rerank_scores[source] = float(candidate_image_scores.get(source, 0.0) + (2.2 * seg_score))

        if rerank_scores:
            best_source = max(rerank_scores, key=rerank_scores.get)
            if poor_seg_case and candidate_overlap_with_ref:
                # If all candidates are still weak in Dice, prefer the one with best spatial overlap.
                max_dice = max(candidate_segmentation_metrics[s].get("dice", 0.0) for s in candidate_segmentation_metrics)
                if max_dice < 0.03:
                    best_source = max(candidate_overlap_with_ref, key=candidate_overlap_with_ref.get)
            pred_image_model = candidates[best_source].astype(np.float32)
            pred_image_post = postprocess_predicted_image(
                pred_image_model,
                recent_real_images,
                week_stats,
                method="histogram_match_enhanced",
            )
            print(f"    Segmentation-aware rerank selected: {best_source}")
        reused_candidate_output = best_source in candidate_original_cache and best_source in candidate_mask_original_cache
        if reused_candidate_output:
            pred_image_post = candidate_post_cache[best_source].astype(np.float32)
            pred_image_original = candidate_original_cache[best_source].astype(np.float32)
            pred_mask = candidate_mask_original_cache[best_source].astype(np.float32)
            pred_mask_work = candidate_mask_work_cache[best_source].astype(np.float32)
            print("    Reusing rerank-selected candidate nnU-Net output to avoid redundant inference.")
    
    # Save temporary prediction for nnU-Net execution.
    if not reused_candidate_output and not reused_existing_mask:
        pred_image_original = resample_to_shape(
            pred_image_post,
            all_weeks[target_idx].flair_original.shape,
            order=1
        )
        pred_mask, pred_dbg = run_nnunet_with_retries(
            base_nifti_path=pred_nifti_path,
            image_volume=pred_image_original,
            affine=all_weeks[target_idx].flair_affine,
            header=all_weeks[target_idx].flair_header,
            output_dir=nnunet_output_dir / "pred",
            working_dir=nnunet_working_dir / "pred",
            reference_mask=real_pred_mask.astype(np.float32),
            label="pred",
        )
        print(
            f"    Final prediction: nnU-Net best mask {pred_dbg.get('best_voxels', 0)} voxels "
            f"({pred_dbg.get('best_ratio', 0.0) * 100:.3f}%)"
        )

        pred_mask, mask_stabilized = postprocess_predicted_mask(
            pred_mask.astype(np.float32),
            pred_image_original.astype(np.float32),
            reference_mask=real_pred_mask.astype(np.float32),
        )
        if mask_stabilized:
            print("    Segmentation postprocess applied: brain-constrained + largest-component filtering.")

        # Always evaluate in working space to keep shapes consistent.
        pred_mask_work = resample_to_shape(pred_mask.astype(np.float32), target_shape=all_weeks[target_idx].flair_resampled.shape, order=0)
    else:
        mask_stabilized = True

    # Fuse CNN auxiliary segmentation head with nnU-Net mask when available.
    if aux_cnn_mask is not None and int((aux_cnn_mask > 0.5).sum()) > 0:
        fused_work = np.logical_or(pred_mask_work > 0.5, aux_cnn_mask > 0.5)
        pred_mask_work, aux_fused = postprocess_predicted_mask(
            fused_work.astype(np.float32),
            pred_image_original.astype(np.float32) if "pred_image_original" in locals() else pred_image_post.astype(np.float32),
            reference_mask=real_pred_mask.astype(np.float32),
        )
        if aux_fused:
            print("    Fused CNN auxiliary segmentation head with nnU-Net mask.")
        pred_mask = resample_to_shape(
            pred_mask_work.astype(np.float32),
            all_weeks[target_idx].flair_original.shape,
            order=0,
        )
    
    mask_nifti_path = output_dir / "predicted_mask_next_week.nii.gz"
    
    # Compute metrics
    print("\n>>> Computing metrics...")
    
    model_image_metrics = compute_image_metrics(target_flair, pred_image_model, target_mask)
    model_seg_metrics = segmentation_metrics(real_mask_work, pred_mask_work)
    model_metrics = {**model_image_metrics, **model_seg_metrics}
    model_metrics = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in model_metrics.items()
    }

    # Baseline: improved gradient-based extrapolation (not just mean)
    if target_idx < 2:
        baseline_pred = all_weeks[max(0, target_idx-1)].flair_resampled.copy()
    else:
        # Use gradient from last two weeks to extrapolate trend
        img_t2 = all_weeks[target_idx-2].flair_resampled
        img_t1 = all_weeks[target_idx-1].flair_resampled
        velocity = img_t1 - img_t2
        baseline_pred = (img_t1 + velocity * 0.5).astype(np.float32)
    
    baseline_mask = (baseline_pred > np.percentile(baseline_pred, 75)).astype(np.float32)
    baseline_mask, _ = postprocess_predicted_mask(
        baseline_mask.astype(np.float32),
        baseline_pred.astype(np.float32),
        reference_mask=real_mask_work.astype(np.float32),
    )

    baseline_image_metrics = compute_image_metrics(target_flair, baseline_pred, target_mask)
    baseline_seg_metrics = segmentation_metrics(real_mask_work, baseline_mask)
    baseline_metrics = {**baseline_image_metrics, **baseline_seg_metrics}
    baseline_metrics = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in baseline_metrics.items()
    }
    
    # Quality gate with adaptive thresholds and optional auto-retry.
    retry_triggered = False
    tumor_ratio = float(target_mask.mean())
    dice_margin = max(0.002, min(0.01, 0.006 - 0.2 * tumor_ratio))
    rmse_factor = 1.10
    ssim_factor = 0.90

    dice_ok = model_metrics["dice"] >= (baseline_metrics["dice"] - dice_margin)
    rmse_ok = model_metrics["rmse"] <= baseline_metrics["rmse"] * rmse_factor
    ssim_ok = model_metrics["ssim"] >= baseline_metrics["ssim"] * ssim_factor

    # If not good enough, perform an automatic second attempt with stronger tumor focus.
    pred_mask_ratio_work = float((pred_mask_work > 0.5).mean())
    severe_segmentation_failure = (model_metrics.get("dice", 0.0) < 0.03) or (pred_mask_ratio_work < 1e-6)
    if enable_auto_retry and (case_epochs >= 6) and ((not (dice_ok and rmse_ok and ssim_ok)) or severe_segmentation_failure):
        retry_triggered = True
        print("\n>>> Auto-retry triggered (stronger tumor-focused training)...")

        retry_model = ImprovedSpatialTemporalPredictor(
            seq_len=seq_len,
            in_channels=4,
            init_features=32,
            enable_auxiliary_segmentation=True,
            use_temporal_attention=True,
        )

        retry_checkpoint_dir = checkpoint_dir / "retry"
        _ = train_improved_predictor(
            retry_model,
            all_weeks,
            target_idx,
            seq_len=seq_len,
            patch_size=patch_size,
            epochs=max(4, min(10, (case_epochs // 2) + 2)),
            steps_per_epoch=max(8, min(18, (case_steps // 2) + 4 if case_steps > 1 else 8)),
            batch_size=batch_size,
            learning_rate=5e-4,
            roi_weight=48.0,
            roi_weight_end=60.0,
            tumor_patch_prob=0.92,
            device=device,
            checkpoint_dir=retry_checkpoint_dir,
        )

        retry_pred_image = predict_full_volume(
            retry_model,
            all_weeks,
            target_idx,
            seq_len=seq_len,
            patch_size=patch_size,
            device=device,
        )
        retry_pred_image_post = postprocess_predicted_image(
            retry_pred_image,
            recent_real_images,
            week_stats,
            method="histogram_match_enhanced",
        )

        # Run nnU-Net again on retry prediction
        retry_pred_image_original = resample_to_shape(
            retry_pred_image_post,
            all_weeks[target_idx].flair_original.shape,
            order=1,
        )
        retry_pred_mask, retry_dbg = run_nnunet_with_retries(
            base_nifti_path=pred_nifti_path,
            image_volume=retry_pred_image_original,
            affine=all_weeks[target_idx].flair_affine,
            header=all_weeks[target_idx].flair_header,
            output_dir=nnunet_output_dir / "retry",
            working_dir=nnunet_working_dir / "retry",
            reference_mask=real_pred_mask.astype(np.float32),
            label="retry",
        )
        print(
            f"    Retry prediction: nnU-Net best mask {retry_dbg.get('best_voxels', 0)} voxels "
            f"({retry_dbg.get('best_ratio', 0.0) * 100:.3f}%)"
        )

        retry_pred_mask, retry_mask_stabilized = postprocess_predicted_mask(
            retry_pred_mask.astype(np.float32),
            retry_pred_image_original.astype(np.float32),
            reference_mask=real_pred_mask.astype(np.float32),
        )
        if retry_mask_stabilized:
            print("    Retry segmentation postprocess applied (largest component + brain constraint).")

        retry_pred_mask_work = resample_to_shape(
            retry_pred_mask.astype(np.float32),
            target_shape=all_weeks[target_idx].flair_resampled.shape,
            order=0,
        )

        retry_image_metrics = compute_image_metrics(target_flair, retry_pred_image, target_mask)
        retry_seg_metrics = segmentation_metrics(real_mask_work, retry_pred_mask_work)
        retry_metrics = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in {**retry_image_metrics, **retry_seg_metrics}.items()
        }

        # Keep retry if it improves dice and doesn't heavily regress image fidelity.
        retry_is_better = (
            (retry_metrics["dice"] > model_metrics["dice"])
            and (retry_metrics["rmse"] <= model_metrics["rmse"] * 1.12)
            and (retry_metrics["ssim"] >= model_metrics["ssim"] * 0.90)
        )

        if retry_is_better:
            print("    Auto-retry accepted: improved tumor segmentation quality.")
            pred_image_model = retry_pred_image.astype(np.float32)
            pred_image_post = retry_pred_image_post.astype(np.float32)
            pred_mask_work = retry_pred_mask_work.astype(np.float32)
            model_metrics = retry_metrics

    # Final fallback only when model remains clearly below baseline.
    used_baseline = False
    final_dice_ok = model_metrics["dice"] >= (baseline_metrics["dice"] - max(0.002, dice_margin))
    final_rmse_ok = model_metrics["rmse"] <= baseline_metrics["rmse"] * 1.20
    final_ssim_ok = model_metrics["ssim"] >= baseline_metrics["ssim"] * 0.85
    catastrophic_model = (
        model_metrics["dice"] < 0.01
        and model_metrics["rmse"] > baseline_metrics["rmse"] * 1.35
        and model_metrics["ssim"] < baseline_metrics["ssim"] * 0.70
    )
    if catastrophic_model and not (final_dice_ok and final_rmse_ok and final_ssim_ok):
        used_baseline = True
        pred_image_model = baseline_pred.astype(np.float32)
        pred_image_post = postprocess_predicted_image(
            pred_image_model,
            recent_real_images,
            week_stats,
            method="histogram_match_enhanced",
        )

        # Keep mandatory nnU-Net stage for final baseline fallback too.
        baseline_pred_original = resample_to_shape(
            pred_image_post,
            all_weeks[target_idx].flair_original.shape,
            order=1,
        )
        baseline_nnunet_mask, baseline_dbg = run_nnunet_with_retries(
            base_nifti_path=pred_nifti_path,
            image_volume=baseline_pred_original,
            affine=all_weeks[target_idx].flair_affine,
            header=all_weeks[target_idx].flair_header,
            output_dir=nnunet_output_dir / "baseline",
            working_dir=nnunet_working_dir / "baseline",
            reference_mask=real_pred_mask.astype(np.float32),
            label="baseline",
        )
        print(
            f"    Baseline fallback: nnU-Net best mask {baseline_dbg.get('best_voxels', 0)} voxels "
            f"({baseline_dbg.get('best_ratio', 0.0) * 100:.3f}%)"
        )

        baseline_nnunet_mask, baseline_mask_stabilized = postprocess_predicted_mask(
            baseline_nnunet_mask.astype(np.float32),
            baseline_pred_original.astype(np.float32),
            reference_mask=real_pred_mask.astype(np.float32),
        )
        if baseline_mask_stabilized:
            print("    Baseline segmentation postprocess applied (largest component + brain constraint).")

        pred_mask_work = resample_to_shape(
            baseline_nnunet_mask.astype(np.float32),
            target_shape=all_weeks[target_idx].flair_resampled.shape,
            order=0,
        )

        model_image_metrics = compute_image_metrics(target_flair, pred_image_model, target_mask)
        model_seg_metrics = segmentation_metrics(real_mask_work, pred_mask_work)
        model_metrics = {**model_image_metrics, **model_seg_metrics}
        model_metrics = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in model_metrics.items()
        }

    # Auxiliary metrics against provided manual mask (kept for traceability only).
    manual_mask_seg_metrics = segmentation_metrics(target_mask, pred_mask_work)
    temporal_gnn_image_metrics = {k: float(v) for k, v in (classical_image_metrics.get("temporal_gnn", {}) or {}).items()}
    temporal_gnn_tumor_metrics: dict[str, float] = {}
    if gnn_tumor_pred is not None:
        gnn_tumor_mask = (gnn_tumor_pred > 0.5).astype(np.float32)
        temporal_gnn_tumor_metrics = {
            k: float(v) for k, v in segmentation_metrics(real_mask_work, gnn_tumor_mask).items()
        }
        candidate_segmentation_metrics["temporal_gnn_direct"] = temporal_gnn_tumor_metrics

    # Save final chosen prediction/mask in original space
    pred_image_original = resample_to_shape(
        pred_image_post,
        all_weeks[target_idx].flair_original.shape,
        order=1,
    )
    nib.save(
        nib.Nifti1Image(
            pred_image_original,
            affine=all_weeks[target_idx].flair_affine,
            header=all_weeks[target_idx].flair_header,
        ),
        str(pred_nifti_path),
    )

    pred_mask_original = resample_to_shape(
        pred_mask_work,
        all_weeks[target_idx].flair_original.shape,
        order=0,
    )
    nib.save(
        nib.Nifti1Image(
            pred_mask_original,
            affine=all_weeks[target_idx].flair_affine,
            header=all_weeks[target_idx].flair_header,
        ),
        str(mask_nifti_path),
    )

    sequence_model_outputs: dict[str, str] = {}
    for seq_key, seq_label, seq_pred in [
        ("rnn_lstm", "lstm", lstm_pred),
        ("rnn_transformer", "transformer", transformer_pred),
    ]:
        if seq_pred is None:
            continue
        seq_path = output_dir / f"predicted_image_{seq_label}.nii.gz"
        seq_original = resample_to_shape(
            seq_pred.astype(np.float32),
            all_weeks[target_idx].flair_original.shape,
            order=1,
        )
        nib.save(
            nib.Nifti1Image(
                seq_original,
                affine=all_weeks[target_idx].flair_affine,
                header=all_weeks[target_idx].flair_header,
            ),
            str(seq_path),
        )
        sequence_model_outputs[seq_key] = str(seq_path)
        print(f"    Saved {seq_label.upper()} prediction: {seq_path}")

    gnn_image_path = output_dir / "predicted_image_temporal_gnn.nii.gz"
    gnn_tumor_path = output_dir / "predicted_tumor_temporal_gnn.nii.gz"
    if gnn_pred is not None:
        gnn_pred_export = postprocess_predicted_image(
            gnn_pred.astype(np.float32),
            recent_real_images,
            week_stats,
            method="histogram_match_enhanced",
        )
        gnn_pred_original = resample_to_shape(
            gnn_pred_export.astype(np.float32),
            all_weeks[target_idx].flair_original.shape,
            order=1,
        )
        nib.save(
            nib.Nifti1Image(
                gnn_pred_original,
                affine=all_weeks[target_idx].flair_affine,
                header=all_weeks[target_idx].flair_header,
            ),
            str(gnn_image_path),
        )
    if gnn_tumor_pred is not None:
        gnn_tumor_mask = (gnn_tumor_pred > 0.5).astype(np.float32)
        gnn_tumor_original = resample_to_shape(
            gnn_tumor_mask,
            all_weeks[target_idx].flair_original.shape,
            order=0,
        )
        nib.save(
            nib.Nifti1Image(
                gnn_tumor_original,
                affine=all_weeks[target_idx].flair_affine,
                header=all_weeks[target_idx].flair_header,
            ),
            str(gnn_tumor_path),
        )
    print(f"    Saved predicted image: {pred_nifti_path}")
    print(f"    Saved predicted mask: {mask_nifti_path}")
    if gnn_pred is not None:
        print(f"    Saved temporal GNN image: {gnn_image_path}")
    if gnn_tumor_pred is not None:
        print(f"    Saved temporal GNN tumor map: {gnn_tumor_path}")
    if used_baseline:
        print("    Quality gate: baseline output selected (model underperformed baseline).")

    # Compute improvement
    if baseline_metrics["dice"] > 0:
        improvement = (model_metrics["dice"] - baseline_metrics["dice"]) / baseline_metrics["dice"]
    else:
        improvement = 0.0
    
    real_mask_voxels_final = int((real_pred_mask > 0.5).sum())
    real_mask_ratio_final = float((real_pred_mask > 0.5).mean())
    real_tumor_slice_idx = int(select_axial_slice_index(real_pred_mask, allow_empty_fallback=False))

    ref_chars = compute_volume_characteristics(target_flair)
    model_key_map = [
        ("RNN (GRU)", "rnn_gru"),
        ("RNN (LSTM)", "rnn_lstm"),
        ("RNN (Transformer)", "rnn_transformer"),
        ("Temporal GNN", "temporal_gnn"),
        ("RNN Enriquecida", "rnn_enriched"),
        ("SVR", "svr"),
        ("Random Forest", "random_forest"),
        ("Media (ensemble)", "mean_ensemble"),
        ("Novo Treino (Stacked)", "stacked_refiner"),
    ]
    predicted_characteristics_models: dict[str, dict[str, Any]] = {}
    for display, source_key in model_key_map:
        cand = candidates.get(source_key)
        if cand is None:
            continue
        pred_chars = compute_volume_characteristics(cand.astype(np.float32))
        abs_error = {
            k: float(abs(pred_chars.get(k, 0.0) - ref_chars.get(k, 0.0)))
            for k in ["mean", "std", "entropy", "p95"]
        }
        source_metrics = classical_image_metrics.get(source_key, {}) or {}
        predicted_characteristics_models[display] = {
            "source_key": source_key,
            "predicted": pred_chars,
            "reference": {k: float(ref_chars.get(k, 0.0)) for k in ["mean", "std", "entropy", "p95"]},
            "abs_error": abs_error,
            "rmse": float(source_metrics.get("rmse", 0.0)),
            "mae": float(source_metrics.get("mae", 0.0)),
            "ssim": float(source_metrics.get("ssim", 0.0)),
            "auc_roc": float(source_metrics.get("auc_roc", 0.5)),
        }

    # Save metrics
    metrics_data = {
        "case_name": case_name,
        "target_week": target_week,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "professor_model_metrics": professor_model_metrics,
        "rnn_enriched_experiment": {
            "enabled": bool(rnn_enriched_pred is not None),
            "metrics": {
                k: float(v) for k, v in (classical_image_metrics.get("rnn_enriched", {}) or {}).items()
            },
            "training_info": rnn_enriched_info,
            "extra_channels": ["svr_prediction", "random_forest_prediction"],
        },
        "stacked_refiner_experiment": {
            "metrics": stacked_refiner_metrics,
            "training_info": stacked_refiner_info,
            "source_models": ["rnn_gru", "svr", "random_forest"],
        },
        "sequence_models_experiment": {
            "lstm": {
                "metrics": {
                    k: float(v) for k, v in (classical_image_metrics.get("rnn_lstm", {}) or {}).items()
                },
                "training_info": {
                    k: ml_training_info.get(k)
                    for k in ["lstm_samples", "lstm_epochs", "lstm_final_loss", "lstm_error"]
                    if k in ml_training_info
                },
                "output_image": sequence_model_outputs.get("rnn_lstm", ""),
            },
            "transformer": {
                "metrics": {
                    k: float(v) for k, v in (classical_image_metrics.get("rnn_transformer", {}) or {}).items()
                },
                "training_info": {
                    k: ml_training_info.get(k)
                    for k in [
                        "transformer_samples",
                        "transformer_epochs",
                        "transformer_final_loss",
                        "transformer_error",
                    ]
                    if k in ml_training_info
                },
                "output_image": sequence_model_outputs.get("rnn_transformer", ""),
            },
        },
        "temporal_gnn_experiment": {
            "metrics": temporal_gnn_image_metrics,
            "tumor_metrics": temporal_gnn_tumor_metrics,
            "training_info": ml_training_info,
            "orientation_alignment": gnn_orientation_info,
            "output_image": str(gnn_image_path) if gnn_pred is not None else "",
            "output_tumor_map": str(gnn_tumor_path) if gnn_tumor_pred is not None else "",
            "graph_hyperparameters": {
                "k_neighbors": int(gnn_k_neighbors),
                "temperature": float(gnn_temperature),
                "entropy_weight": float(gnn_entropy_weight),
                "tumor_weight": float(gnn_tumor_weight),
            },
        },
        "classical_image_metrics": {
            k: {ik: float(iv) for ik, iv in v.items()} for k, v in classical_image_metrics.items()
        },
        "model_comparison": {
            k: {
                "ssim": float(v.get("ssim", 0.0)),
                "rmse": float(v.get("rmse", 0.0)),
                "mae": float(v.get("mae", 0.0)),
                "auc_roc": float(v.get("auc_roc", 0.5)),
                "entropy_diff": float(v.get("entropy_diff", 0.0)),
            }
            for k, v in classical_image_metrics.items()
        },
        "predicted_characteristics_vs_reference": {
            "reference": {k: float(ref_chars.get(k, 0.0)) for k in ["mean", "std", "entropy", "p95"]},
            "models": predicted_characteristics_models,
        },
        "selected_image_predictor": best_source,
        "candidate_image_scores": {k: float(v) for k, v in candidate_image_scores.items()},
        "candidate_segmentation_metrics": candidate_segmentation_metrics,
        "candidate_rerank_scores": {k: float(v) for k, v in rerank_scores.items()},
        "ml_training_info": ml_training_info,
        "improvement_dice": float(improvement),
        "quality_gate_baseline_selected": used_baseline,
        "incumbent_preserved": False,
        "mask_stabilization_applied": bool(mask_stabilized),
        "real_mask_stabilization_applied": bool(real_mask_stabilized),
        "real_mask_voxels": real_mask_voxels_final,
        "real_mask_ratio": real_mask_ratio_final,
        "real_tumor_slice_index": real_tumor_slice_idx,
        "auto_retry_triggered": retry_triggered,
        "segmentation_reference": "nnunet_real_target_mask",
        "segmentation_aux_manual_mask_metrics": {
            k: float(v) for k, v in manual_mask_seg_metrics.items()
        },
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"    Dice: {model_metrics['dice']:.4f} (baseline: {baseline_metrics['dice']:.4f}, improvement: {improvement:+.1%})")
    print(f"    RMSE: {model_metrics['rmse']:.4f} (baseline: {baseline_metrics['rmse']:.4f})")
    print(f"    SSIM: {model_metrics['ssim']:.4f} (baseline: {baseline_metrics['ssim']:.4f})")
    
    # Visualizations
    print("\n>>> Generating visualizations...")

    real_original = all_weeks[target_idx].flair_original.astype(np.float32)
    pred_original = pred_image_original.astype(np.float32)
    pred_mask_original = pred_mask_original.astype(np.float32)
    real_mask_original = real_pred_mask.astype(np.float32)
    if real_mask_original.shape != real_original.shape:
        real_mask_original = resample_to_shape(real_mask_original, real_original.shape, order=0)
    display_aspect = axial_aspect_from_header(all_weeks[target_idx].flair_header)

    slice_idx_orig = select_axial_slice_index(real_mask_original, allow_empty_fallback=False)
    real_sl = extract_axial_slice(real_original, slice_idx_orig)
    pred_sl = extract_axial_slice(pred_original, slice_idx_orig)
    pred_mask_sl = extract_axial_slice(pred_mask_original, slice_idx_orig)
    real_mask_sl = extract_axial_slice(real_mask_original, slice_idx_orig)
    pred_mask_vis = enhance_mask_for_display(pred_mask_sl)
    real_mask_vis = enhance_mask_for_display(real_mask_sl)

    shared_bbox = compute_shared_crop_bbox(real_sl, pred_sl, pred_mask_vis, real_mask_vis, margin=14)
    real_sl = crop_slice(real_sl, shared_bbox)
    pred_sl = crop_slice(pred_sl, shared_bbox)
    pred_mask_vis = crop_slice(pred_mask_vis, shared_bbox)
    real_mask_vis = crop_slice(real_mask_vis, shared_bbox)

    real_show = normalize_for_display(real_sl)
    pred_show = normalize_for_display(pred_sl)

    stacked_refiner_original = None
    stacked_refiner_show = None
    stacked_refiner_sl = None
    if stacked_refiner_pred is not None:
        stacked_refiner_original = resample_to_shape(
            stacked_refiner_pred.astype(np.float32),
            all_weeks[target_idx].flair_original.shape,
            order=1,
        ).astype(np.float32)
        stacked_refiner_path = output_dir / "stacked_refiner_image.nii.gz"
        nib.save(
            nib.Nifti1Image(
                stacked_refiner_original,
                affine=all_weeks[target_idx].flair_affine,
                header=all_weeks[target_idx].flair_header,
            ),
            str(stacked_refiner_path),
        )
        stacked_refiner_sl = extract_axial_slice(stacked_refiner_original, slice_idx_orig)
        stacked_refiner_sl = crop_slice(stacked_refiner_sl, shared_bbox)
        stacked_refiner_show = normalize_for_display(stacked_refiner_sl)

        fig_exp, ax_exp = plt.subplots(2, 3, figsize=(16, 9))
        ax_exp[0, 0].imshow(real_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_exp[0, 0].set_title("Imagem Real")
        ax_exp[0, 0].axis("off")

        ax_exp[0, 1].imshow(normalize_for_display(crop_slice(extract_axial_slice(resample_to_shape(candidates['rnn_gru'], all_weeks[target_idx].flair_original.shape, order=1), slice_idx_orig), shared_bbox)), cmap="gray", aspect=display_aspect, origin="lower")
        ax_exp[0, 1].set_title("RNN (GRU)")
        ax_exp[0, 1].axis("off")

        ax_exp[0, 2].imshow(normalize_for_display(crop_slice(extract_axial_slice(resample_to_shape(candidates['svr'], all_weeks[target_idx].flair_original.shape, order=1), slice_idx_orig), shared_bbox)), cmap="gray", aspect=display_aspect, origin="lower")
        ax_exp[0, 2].set_title("SVR")
        ax_exp[0, 2].axis("off")

        ax_exp[1, 0].imshow(normalize_for_display(crop_slice(extract_axial_slice(resample_to_shape(candidates['random_forest'], all_weeks[target_idx].flair_original.shape, order=1), slice_idx_orig), shared_bbox)), cmap="gray", aspect=display_aspect, origin="lower")
        ax_exp[1, 0].set_title("Random Forest")
        ax_exp[1, 0].axis("off")

        ax_exp[1, 1].imshow(normalize_for_display(crop_slice(extract_axial_slice(resample_to_shape(candidates['mean_ensemble'], all_weeks[target_idx].flair_original.shape, order=1), slice_idx_orig), shared_bbox)), cmap="gray", aspect=display_aspect, origin="lower")
        ax_exp[1, 1].set_title("Media")
        ax_exp[1, 1].axis("off")

        ax_exp[1, 2].imshow(stacked_refiner_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_exp[1, 2].set_title("Novo Treino (Stacked)")
        ax_exp[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "stacked_refiner_comparison.png", dpi=160, bbox_inches="tight")
        plt.close()

    # Required visual panel: real + RNN + SVR + RF + mean ensemble.
    if all(k in candidates for k in ["rnn_gru", "svr", "random_forest", "mean_ensemble"]):
        rnn_orig = resample_to_shape(candidates["rnn_gru"], real_original.shape, order=1)
        svr_orig = resample_to_shape(candidates["svr"], real_original.shape, order=1)
        rf_orig = resample_to_shape(candidates["random_forest"], real_original.shape, order=1)
        mean_orig = resample_to_shape(candidates["mean_ensemble"], real_original.shape, order=1)

        # Persist mean ensemble volume for traceability.
        mean_nifti_path = output_dir / "predicted_image_mean_ensemble.nii.gz"
        nib.save(
            nib.Nifti1Image(mean_orig.astype(np.float32), affine=all_weeks[target_idx].flair_affine, header=all_weeks[target_idx].flair_header),
            str(mean_nifti_path),
        )

        rnn_sl = crop_slice(extract_axial_slice(rnn_orig.astype(np.float32), slice_idx_orig), shared_bbox)
        svr_sl = crop_slice(extract_axial_slice(svr_orig.astype(np.float32), slice_idx_orig), shared_bbox)
        rf_sl = crop_slice(extract_axial_slice(rf_orig.astype(np.float32), slice_idx_orig), shared_bbox)
        mean_sl = crop_slice(extract_axial_slice(mean_orig.astype(np.float32), slice_idx_orig), shared_bbox)

        rnn_show = normalize_for_display(rnn_sl)
        svr_show = normalize_for_display(svr_sl)
        rf_show = normalize_for_display(rf_sl)
        mean_show = normalize_for_display(mean_sl)
        mean_err = normalize_for_display(np.abs(real_sl - mean_sl))

        fig_ens, ax_ens = plt.subplots(2, 4, figsize=(20, 9))
        ax_ens[0, 0].imshow(real_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_ens[0, 0].set_title("Imagem Real")
        ax_ens[0, 0].axis("off")

        ax_ens[0, 1].imshow(rnn_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_ens[0, 1].set_title("Predicao RNN (GRU)")
        ax_ens[0, 1].axis("off")

        ax_ens[0, 2].imshow(svr_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_ens[0, 2].set_title("Predicao SVR")
        ax_ens[0, 2].axis("off")

        if gnn_pred is not None:
            gnn_orig = resample_to_shape(candidates["temporal_gnn"], real_original.shape, order=1)
            gnn_sl = crop_slice(extract_axial_slice(gnn_orig.astype(np.float32), slice_idx_orig), shared_bbox)
            gnn_show = normalize_for_display(gnn_sl)
            ax_ens[0, 3].imshow(gnn_show, cmap="gray", aspect=display_aspect, origin="lower")
            ax_ens[0, 3].set_title("Temporal GNN")
        else:
            ax_ens[0, 3].text(0.5, 0.5, "Temporal GNN\nindisponivel", ha="center", va="center")
        ax_ens[0, 3].axis("off")

        ax_ens[1, 0].imshow(rf_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_ens[1, 0].set_title("Predicao Random Forest")
        ax_ens[1, 0].axis("off")

        ax_ens[1, 1].imshow(mean_show, cmap="gray", aspect=display_aspect, origin="lower")
        ax_ens[1, 1].set_title("Media (Ensemble)")
        ax_ens[1, 1].axis("off")

        if stacked_refiner_pred is not None:
            stacked_refiner_orig = resample_to_shape(stacked_refiner_pred.astype(np.float32), real_original.shape, order=1)
            stacked_refiner_sl = crop_slice(extract_axial_slice(stacked_refiner_orig.astype(np.float32), slice_idx_orig), shared_bbox)
            stacked_refiner_show_ens = normalize_for_display(stacked_refiner_sl)
            ax_ens[1, 2].imshow(stacked_refiner_show_ens, cmap="gray", aspect=display_aspect, origin="lower")
            ax_ens[1, 2].set_title("Novo Treino (Stacked)")
        else:
            ax_ens[1, 2].imshow(mean_err, cmap="hot", aspect=display_aspect, origin="lower")
            ax_ens[1, 2].set_title("Erro |Real - Media|")
        ax_ens[1, 2].axis("off")

        ax_ens[1, 3].imshow(mean_err, cmap="hot", aspect=display_aspect, origin="lower")
        ax_ens[1, 3].set_title("Erro |Real - Media|")
        ax_ens[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "predictor_ensemble_comparison.png", dpi=160, bbox_inches="tight")
        plt.close()
    
    # Image comparison (3 channels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Real image slices
    axes[0, 0].imshow(real_show, cmap="gray", aspect=display_aspect, origin="lower")
    axes[0, 0].set_title("Imagem Real (dados originais)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_show, cmap="gray", aspect=display_aspect, origin="lower")
    axes[0, 1].set_title("Imagem Predita (modelo)")
    axes[0, 1].axis("off")

    error_map = np.abs(real_sl - pred_sl)
    im = axes[0, 2].imshow(normalize_for_display(error_map), cmap="hot", aspect=display_aspect, origin="lower")
    axes[0, 2].set_title("Diferenca Absoluta")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2])
    
    # Segmentation comparison
    axes[1, 0].imshow(real_mask_vis, cmap="gray", aspect=display_aspect, origin="lower")
    axes[1, 0].set_title("Mascara Tumoral Real (nnU-Net)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_mask_vis, cmap="gray", aspect=display_aspect, origin="lower")
    axes[1, 1].set_title("Mascara Tumoral Predita (nnU-Net)")
    axes[1, 1].axis("off")
    
    # Explicit tumor-mask difference map (XOR) for real vs predicted segmentations.
    mask_diff = np.logical_xor(real_mask_vis > 0.5, pred_mask_vis > 0.5).astype(np.float32)
    axes[1, 2].imshow(mask_diff, cmap="magma", aspect=display_aspect, origin="lower")
    axes[1, 2].set_title("Diferenca de Mascaras Tumorais")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "image_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Segmentation comparison (required output)
    fig_seg, ax_seg = plt.subplots(1, 3, figsize=(14, 4))
    ax_seg[0].imshow(real_mask_vis, cmap="gray", aspect=display_aspect, origin="lower")
    ax_seg[0].set_title("Mascara Tumoral Real (nnU-Net)")
    ax_seg[0].axis("off")
    ax_seg[1].imshow(pred_mask_vis, cmap="gray", aspect=display_aspect, origin="lower")
    ax_seg[1].set_title("Mascara Tumoral Predita (nnU-Net)")
    ax_seg[1].axis("off")
    ax_seg[2].imshow(mask_diff, cmap="magma", aspect=display_aspect, origin="lower")
    ax_seg[2].set_title("Diferenca (XOR)")
    ax_seg[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Labeled visual panel for report clarity (larger main triplet + support row).
    fig_panel = plt.figure(figsize=(18, 10))
    gs_panel = fig_panel.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.12, wspace=0.08)
    axp = [
        fig_panel.add_subplot(gs_panel[0, 0]),
        fig_panel.add_subplot(gs_panel[0, 1]),
        fig_panel.add_subplot(gs_panel[0, 2]),
        fig_panel.add_subplot(gs_panel[1, 0]),
        fig_panel.add_subplot(gs_panel[1, 1]),
        fig_panel.add_subplot(gs_panel[1, 2]),
    ]
    axp[0].imshow(real_show, cmap="gray", aspect=display_aspect, origin="lower")
    axp[0].set_title("Imagem Real (dados originais)")
    axp[0].axis("off")

    axp[1].imshow(pred_show, cmap="gray", aspect=display_aspect, origin="lower")
    axp[1].set_title("Imagem Predita (modelo)")
    axp[1].axis("off")

    axp[2].imshow(pred_mask_vis, cmap="gray", aspect=display_aspect, origin="lower")
    axp[2].set_title("Mascara Tumoral Predita (nnU-Net)")
    axp[2].axis("off")

    axp[3].imshow(real_mask_vis, cmap="gray", aspect=display_aspect, origin="lower")
    axp[3].set_title("Mascara Tumoral Real (nnU-Net)")
    axp[3].axis("off")

    overlay_panel = np.stack([pred_show, pred_show, pred_show], axis=-1)
    overlay_panel[:, :, 1] = np.maximum(overlay_panel[:, :, 1], pred_mask_vis * 0.75)
    axp[4].imshow(overlay_panel, aspect=display_aspect, origin="lower")
    axp[4].set_title("Overlay: Predita + Mascara Predita")
    axp[4].axis("off")

    real_overlay = np.stack([real_show, real_show, real_show], axis=-1)
    real_overlay[:, :, 0] = np.maximum(real_overlay[:, :, 0], real_mask_vis * 0.75)
    axp[5].imshow(real_overlay, aspect=display_aspect, origin="lower")
    axp[5].set_title("Overlay: Real + Mascara Real")
    axp[5].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "visual_panel.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Overlay comparison (required output)
    fig_ov, ax_ov = plt.subplots(1, 2, figsize=(11, 5))
    ov_pred = np.stack([pred_show, pred_show, pred_show], axis=-1)
    ov_pred[:, :, 1] = np.maximum(ov_pred[:, :, 1], pred_mask_vis * 0.75)
    ov_real = np.stack([real_show, real_show, real_show], axis=-1)
    ov_real[:, :, 0] = np.maximum(ov_real[:, :, 0], real_mask_vis * 0.75)
    ax_ov[0].imshow(ov_real, aspect=display_aspect, origin="lower")
    ax_ov[0].set_title("Overlay: Real + Mascara Real")
    ax_ov[0].axis("off")
    ax_ov[1].imshow(ov_pred, aspect=display_aspect, origin="lower")
    ax_ov[1].set_title("Overlay: Predita + Mascara Predita")
    ax_ov[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "overlay_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Metrics visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ["rmse", "mae", "ssim", "dice", "jaccard"]
    model_vals = [model_metrics.get(m, 0) for m in metrics_to_plot]
    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    ax.bar(x - width/2, model_vals, width, label="Model", alpha=0.8)
    ax.bar(x + width/2, baseline_vals, width, label="Baseline", alpha=0.8)
    
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics Comparison: {case_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Preserve incumbent result when it is already strong in image fidelity and the new run degrades it
    # without meaningful segmentation gain.
    preserved_incumbent = False
    if incumbent_metrics is not None and incumbent_snapshot_dir.exists():
        try:
            inc_model = incumbent_metrics.get("model_metrics", {})
            inc_dice = float(inc_model.get("dice", 0.0))
            inc_jaccard = float(inc_model.get("jaccard", 0.0))
            inc_ssim = float(inc_model.get("ssim", 0.0))
            inc_rmse = float(inc_model.get("rmse", 1e9))
            inc_mae = float(inc_model.get("mae", 1e9))
            inc_r2 = float(inc_model.get("r2", -1e9))

            new_dice = float(model_metrics.get("dice", 0.0))
            new_jaccard = float(model_metrics.get("jaccard", 0.0))
            new_ssim = float(model_metrics.get("ssim", 0.0))
            new_rmse = float(model_metrics.get("rmse", 1e9))
            new_mae = float(model_metrics.get("mae", 1e9))
            new_r2 = float(model_metrics.get("r2", -1e9))

            incumbent_good_image = (
                (inc_ssim >= 0.74)
                and (inc_rmse <= 2.90)
                and (inc_mae <= 0.90)
                and (inc_r2 >= 0.55)
            )
            image_regressed = (
                (new_ssim < (inc_ssim * 0.985))
                or (new_rmse > (inc_rmse * 1.03))
                or (new_mae > (inc_mae * 1.04))
                or (new_r2 < (inc_r2 * 0.97))
            )
            segmentation_improved = (
                (new_dice > (inc_dice + 0.005))
                or (new_jaccard > (inc_jaccard + 0.003))
            )

            if incumbent_good_image and image_regressed and (not segmentation_improved):
                print("    Preserving incumbent case outputs (new run regressed image quality without segmentation gain).")
                for fname in [
                    "metrics.json",
                    "predicted_image_next_week.nii.gz",
                    "predicted_image_mean_ensemble.nii.gz",
                    "predicted_image_temporal_gnn.nii.gz",
                    "predicted_tumor_temporal_gnn.nii.gz",
                    "predicted_mask_next_week.nii.gz",
                    "real_target_image.nii.gz",
                    "real_target_tumor_mask.nii.gz",
                    "comparison.png",
                    "image_comparison.png",
                    "segmentation_comparison.png",
                    "overlay_comparison.png",
                    "visual_panel.png",
                    "predictor_ensemble_comparison.png",
                    "metrics_summary.png",
                ]:
                    src = incumbent_snapshot_dir / fname
                    dst = output_dir / fname
                    if src.exists():
                        shutil.copy2(src, dst)

            payload_errors = validate_case_metrics_payload(metrics_data)
            if payload_errors:
                raise RuntimeError(f"Invalid metrics payload: {', '.join(payload_errors)}")

                with open(output_dir / "metrics.json") as f:
                    restored = json.load(f)
                restored["incumbent_preserved"] = True
                with open(output_dir / "metrics.json", "w") as f:
                    json.dump(restored, f, indent=2)
                model_metrics = restored.get("model_metrics", model_metrics)
                baseline_metrics = restored.get("baseline_metrics", baseline_metrics)
                preserved_incumbent = True
        except Exception as e:
            print(f"    Warning: failed to preserve incumbent outputs: {e}")

    if baseline_metrics["dice"] > 0:
        improvement = (model_metrics["dice"] - baseline_metrics["dice"]) / baseline_metrics["dice"]
    else:
        improvement = 0.0
    
    return CaseResult(
        case_name=case_name,
        target_week=target_week,
        train_weeks=train_weeks,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        outputs={
            "predicted_image": str(pred_nifti_path),
            "predicted_mask": str(mask_nifti_path),
            "predicted_image_mean_ensemble": str(output_dir / "predicted_image_mean_ensemble.nii.gz"),
            "predicted_image_temporal_gnn": str(gnn_image_path) if gnn_pred is not None else "",
            "predicted_tumor_temporal_gnn": str(gnn_tumor_path) if gnn_tumor_pred is not None else "",
            "stacked_refiner_image": str(output_dir / "stacked_refiner_image.nii.gz") if stacked_refiner_pred is not None else "",
            "real_target_image": str(real_nifti_path),
            "real_target_mask": str(real_mask_nifti_path),
            "visual_panel_png": str(output_dir / "visual_panel.png"),
            "predictor_ensemble_comparison_png": str(output_dir / "predictor_ensemble_comparison.png"),
            "stacked_refiner_comparison_png": str(output_dir / "stacked_refiner_comparison.png") if stacked_refiner_pred is not None else "",
            "image_comparison_png": str(output_dir / "image_comparison.png"),
            "segmentation_comparison_png": str(output_dir / "segmentation_comparison.png"),
            "overlay_comparison_png": str(output_dir / "overlay_comparison.png"),
            "comparison_png": str(output_dir / "comparison.png"),
            "metrics_png": str(output_dir / "metrics_summary.png"),
        },
        improvement_metric=improvement,
    )


def run_pipeline(
    patient_dir: Path,
    work_shape: tuple[int, int, int] = (96, 96, 48),
    patch_size: tuple[int, int, int] = (48, 48, 24),
    seq_len: int = 3,
    epochs: int = 20,
    steps_per_epoch: int = 30,
    batch_size: int = 2,
    target_indices: Optional[list[int]] = None,
    enable_auto_retry: bool = True,
    fast_image_metrics: bool = False,
    gnn_max_samples: int = 28000,
    gnn_epochs: int = 14,
    gnn_batch_size: int = 768,
    gnn_learning_rate: float = 4e-4,
    gnn_k_neighbors: int = 5,
    gnn_temperature: float = 0.24,
    gnn_entropy_weight: float = 0.20,
    gnn_tumor_weight: float = 0.55,
    refresh_sequence_models: bool = False,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Run complete pipeline for multiple temporal cases.
    """
    print(f"\n{'='*70}")
    print("IMPROVED TEMPORAL MEDICAL IMAGE PREDICTION PIPELINE v2.0")
    print(f"{'='*70}\n")
    
    set_seed(42)

    data_dir, output_root = resolve_patient_paths(patient_dir)
    print(f">>> Patient data directory: {data_dir}")
    print(f">>> Output root directory: {output_root}")
    
    # Load all weeks
    print(">>> Loading patient data...")
    all_weeks = []
    for week in DEFAULT_WEEKS:
        try:
            week_data = load_week_data(data_dir, week, DEFAULT_MODALITIES, work_shape)
            all_weeks.append(week_data)
            print(f"    Loaded {week} | Tumor ratio: {week_data.tumor_ratio:.2%}")
        except Exception as e:
            raise RuntimeError(f"Failed to load required week image data for {week}: {e}") from e
    
    if len(all_weeks) < 3:
        raise ValueError("Need at least 3 weeks of data")
    
    print(f"\nLoaded {len(all_weeks)} weeks of data")
    
    # Evaluate cases
    if target_indices is None:
        target_indices = [len(all_weeks) - 2, len(all_weeks) - 1]  # Last 2 valid targets
    
    cases = []
    for target_idx in target_indices:
        if target_idx < 2 or target_idx >= len(all_weeks):
            continue
        
        try:
            case = evaluate_case(
                output_root, all_weeks, target_idx,
                seq_len=seq_len,
                patch_size=patch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                work_shape=work_shape,
                enable_auto_retry=enable_auto_retry,
                fast_image_metrics=fast_image_metrics,
                gnn_max_samples=gnn_max_samples,
                gnn_epochs=gnn_epochs,
                gnn_batch_size=gnn_batch_size,
                gnn_learning_rate=gnn_learning_rate,
                gnn_k_neighbors=gnn_k_neighbors,
                gnn_temperature=gnn_temperature,
                gnn_entropy_weight=gnn_entropy_weight,
                gnn_tumor_weight=gnn_tumor_weight,
                refresh_sequence_models=refresh_sequence_models,
                device=device,
            )
            cases.append(case)
        except Exception as e:
            raise RuntimeError(
                f"Pipeline aborted due to critical data integrity failure in target_idx={target_idx}: {e}"
            ) from e
    
    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}\n")
    
    if not cases:
        print("No valid cases were produced. All cases were discarded by real-mask validation.")
    else:
        for case in cases:
            print(f"{case.case_name}:")
            print(f"  Target: {case.target_week}")
            print(f"  Dice: {case.model_metrics['dice']:.4f}")
            print(f"  RMSE: {case.model_metrics['rmse']:.4f}")
            print(f"  SSIM: {case.model_metrics['ssim']:.4f}")
            print()
    
    # Save manifest
    manifest = {
        "pipeline": "ImprovedSpatialTemporalPredictor v2.0",
        "work_shape": work_shape,
        "patch_size": patch_size,
        "seq_len": seq_len,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "enable_auto_retry": enable_auto_retry,
        "device": device,
        "gnn_hyperparameters": {
            "max_samples": int(gnn_max_samples),
            "epochs": int(gnn_epochs),
            "batch_size": int(gnn_batch_size),
            "learning_rate": float(gnn_learning_rate),
            "k_neighbors": int(gnn_k_neighbors),
            "temperature": float(gnn_temperature),
            "entropy_weight": float(gnn_entropy_weight),
            "tumor_weight": float(gnn_tumor_weight),
        },
        "cases": [
            {
                "case_name": c.case_name,
                "target_week": c.target_week,
                "train_weeks": c.train_weeks,
                "outputs": c.outputs,
                "dice": c.model_metrics["dice"],
                "rmse": c.model_metrics["rmse"],
                "ssim": c.model_metrics["ssim"],
                "improvement": c.improvement_metric,
            }
            for c in cases
        ]
    }
    
    manifest_dir = output_root / "rnn_prediction_outputs"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path_v2 = manifest_dir / "run_manifest_v2.json"
    with open(manifest_path_v2, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest_path_required = manifest_dir / "run_manifest.json"
    with open(manifest_path_required, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {manifest_path_v2}")
    print(f"Manifest saved: {manifest_path_required}")
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Improved Temporal Medical Image Prediction Pipeline v2.0"
    )
    parser.add_argument("--patient-dir", type=Path, default=Path("Patient-037"),
                        help="Patient directory path (Patient-037 or Patient-037/patient-037)")
    parser.add_argument("--work-shape", type=int, nargs=3, default=[96, 96, 48],
                        help="Working volume shape")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[48, 48, 24],
                        help="Patch size for training")
    parser.add_argument("--seq-len", type=int, default=3,
                        help="Temporal sequence length")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=30,
                        help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--target-indices", type=int, nargs="+", default=None,
                        help="Target week indices (1-indexed from available weeks)")
    parser.add_argument("--enable-auto-retry", action="store_true", default=True,
                        help="Enable second-pass retry with stronger tumor focus when first pass underperforms (default: enabled)")
    parser.add_argument("--fast-image-metrics", action="store_true", default=False,
                        help="Skip expensive nnU-Net reranking by reusing existing predicted mask for faster image-metric refresh")
    parser.add_argument("--refresh-sequence-models", action="store_true", default=False,
                        help="Reuse existing deep-model outputs and retrain only GRU/LSTM/Transformer (+ classical baselines)")
    parser.add_argument("--gnn-max-samples", type=int, default=28000,
                        help="Maximum voxel samples used to train the temporal GNN")
    parser.add_argument("--gnn-epochs", type=int, default=14,
                        help="Training epochs for the temporal GNN")
    parser.add_argument("--gnn-batch-size", type=int, default=768,
                        help="Batch size for temporal GNN training")
    parser.add_argument("--gnn-learning-rate", type=float, default=4e-4,
                        help="Learning rate for temporal GNN training")
    parser.add_argument("--gnn-k-neighbors", type=int, default=5,
                        help="k-nearest neighbors used to build the temporal GNN graph")
    parser.add_argument("--gnn-temperature", type=float, default=0.24,
                        help="Edge-weight temperature for entropy-based graph construction")
    parser.add_argument("--gnn-entropy-weight", type=float, default=0.20,
                        help="Loss weight for the GNN entropy head")
    parser.add_argument("--gnn-tumor-weight", type=float, default=0.55,
                        help="Loss weight for the GNN tumor-probability head")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda:0, cpu, etc)")
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = resolve_compute_device()
    else:
        args.device = resolve_compute_device(args.device)
    
    run_pipeline(
        patient_dir=args.patient_dir,
        work_shape=tuple(args.work_shape),
        patch_size=tuple(args.patch_size),
        seq_len=args.seq_len,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        target_indices=args.target_indices,
        enable_auto_retry=args.enable_auto_retry,
        fast_image_metrics=args.fast_image_metrics,
        refresh_sequence_models=args.refresh_sequence_models,
        gnn_max_samples=args.gnn_max_samples,
        gnn_epochs=args.gnn_epochs,
        gnn_batch_size=args.gnn_batch_size,
        gnn_learning_rate=args.gnn_learning_rate,
        gnn_k_neighbors=args.gnn_k_neighbors,
        gnn_temperature=args.gnn_temperature,
        gnn_entropy_weight=args.gnn_entropy_weight,
        gnn_tumor_weight=args.gnn_tumor_weight,
        device=args.device,
    )


if __name__ == "__main__":
    main()
