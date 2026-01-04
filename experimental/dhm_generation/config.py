"""
Configuration for Alphabet Generation Experiment

This module defines all experimental parameters and constants.
"""

from dataclasses import dataclass
from typing import Literal

# 実験モード
DiffusionMode = Literal["ddim", "ddpm"]

# ベクトル空間の次元数オプション
VALID_DIMENSIONS = [512, 1024, 2048]


@dataclass
class ExperimentConfig:
    """実験設定"""
    
    # ベクトル空間の次元数
    dimension: int = 2048
    
    # 拡散モード
    mode: DiffusionMode = "ddim"
    
    # 拡散ステップ数
    num_timesteps: int = 50
    
    # ノイズスケール
    noise_scale: float = 1.0
    
    # ランダムシード（再現性のため）
    random_seed: int = 42
    
    # アルファスケジュール（線形）
    alpha_start: float = 0.9999
    alpha_end: float = 0.0001
    
    # 正規化の許容誤差
    norm_tolerance: float = 1e-6
    
    # 共鳴マージンの閾値
    resonance_margin_threshold: float = 0.1
    
    def __post_init__(self):
        """設定の検証"""
        if self.dimension not in VALID_DIMENSIONS:
            raise ValueError(
                f"dimension must be one of {VALID_DIMENSIONS}, got {self.dimension}"
            )
        
        if self.mode not in ["ddim", "ddpm"]:
            raise ValueError(f"mode must be 'ddim' or 'ddpm', got {self.mode}")
        
        if self.num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {self.num_timesteps}")
        
        if self.noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {self.noise_scale}")


# デフォルト設定
DEFAULT_CONFIG = ExperimentConfig()
