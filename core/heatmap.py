import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_controlled_prob_field(X_km, Y_km, hotspots, beta=2.2, alpha=0.2):
    """
    生成可控的多热点概率场
    结合了真实感背景噪声(1/f^beta)和自定义的高斯热点
    
    参数:
    X_km, Y_km: 网格坐标矩阵
    hotspots: 字典列表，控制热点属性
    beta: 背景噪声频谱指数 (默认 2.2)
    alpha: 背景噪声的权重 (0~1)。设为0则完全没有环境杂波，只有纯净热点；设为0.2~0.4比较逼真。
    """
    ny, nx = X_km.shape

    # ---- 1) 生成用户控制的多高斯热点 (P_bumps) ----
    P_bumps = np.zeros((ny, nx))
    
    for hs in hotspots:
        cx, cy = hs['center']
        sig_x, sig_y = hs['sigma']
        weight = hs.get('weight', 1.0)
        
        # 叠加二维高斯分布
        bump = weight * np.exp(-((X_km - cx)**2 / (2 * sig_x**2) + (Y_km - cy)**2 / (2 * sig_y**2)))
        P_bumps += bump
        
    # 局部高斯峰值归一化
    if np.max(P_bumps) > 0:
        P_bumps = P_bumps / np.max(P_bumps)

    # ---- 2) 生成逼真背景噪声 (P_field) ----
    W = np.random.randn(ny, nx)
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    FX, FY = np.meshgrid(fx, fy)
    K = np.sqrt(FX**2 + FY**2)
    K[0, 0] = np.min(K[K > 0]) # 避免除零
    
    H = 1.0 / (K**beta)
    F = np.fft.fft2(W) * H
    G = np.real(np.fft.ifft2(F))
    
    G = G - np.min(G)
    if np.max(G) > 0:
        G = G / np.max(G)
    P_field = G

    # ---- 3) 融合与增强 ----
    # 按照 alpha 权重融合背景噪声和自定义热点
    P = alpha * P_field + (1 - alpha) * P_bumps
    P = np.sqrt(P) + 0.05        # 根号增强对比度，扩散 & 加底噪
    P = np.maximum(P, 0)         # 确保非负
    
    # 归一化成全局概率密度网格
    prob_grid = P / (np.sum(P) + np.finfo(float).eps) 
    return prob_grid
