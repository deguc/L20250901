#%%
import numpy as np
import matplotlib.pyplot as plt

# ===== MWC model =====
# n: サブユニット数（Hbなら4）
# KR, KT: R/T状態での解離定数 [mmHg]（小さいほど高親和性）
# L0: T/R 平衡定数（CO2↑やpH↓で大きくする＝T安定化）
def mwc_mean_bound(PO2, n=4, KR=3.0, KT=150.0, L0=900.0):
    PO2 = np.asarray(PO2, dtype=float)
    aR = PO2 / KR
    aT = PO2 / KT
    num = n * (aR*(1+aR)**(n-1) + L0 * aT*(1+aT)**(n-1))
    den = (1+aR)**n + L0 * (1+aT)**n
    N = num / den              # 平均結合数（0〜n）
    Y = N / n                  # 飽和度（fractional saturation）
    return N, Y

# ===== Bohr効果（CO2 & pH で L0 を変える簡易モデル） =====
# 参照値からのズレを  L0_eff = L0_base * exp( beta_CO2*(PCO2-P0) - alpha_pH*(pH-pH0) )
# として反映。係数はチューニング用（kBT単位相当の無次元パラメータ）。
def L0_with_bohr(PCO2, pH, L0_base=900.0, P0=40.0, pH0=7.40,
                 beta_CO2=0.03, alpha_pH=5.0):
    return L0_base * np.exp(beta_CO2*(PCO2 - P0) - alpha_pH*(pH - pH0))

# ===== デモ：PO2-飽和曲線（CO2 と pH でシフト） =====
PO2 = np.logspace(-1, 2.5, 400)   # 0.1〜~316 mmHg
n = 4
KR, KT = 3.0, 150.0               # 親和性（R高・T低）→必要に応じ調整
L0_base = 900.0                   # 安静時のT優勢度

# いくつかの条件で描画
scenarios = [
    {"PCO2": 40.0, "pH": 7.40, "label": "PCO2=40, pH=7.40 (基準)"},
    {"PCO2": 60.0, "pH": 7.40, "label": "PCO2=60, pH=7.40 (高CO2)"},
    {"PCO2": 20.0, "pH": 7.40, "label": "PCO2=20, pH=7.40 (低CO2)"},
    {"PCO2": 40.0, "pH": 7.20, "label": "PCO2=40, pH=7.20 (酸性)"},
    {"PCO2": 40.0, "pH": 7.60, "label": "PCO2=40, pH=7.60 (アルカリ)"},
]

plt.figure(figsize=(6,4.5))
for sc in scenarios:
    L0_eff = L0_with_bohr(sc["PCO2"], sc["pH"], L0_base=L0_base)
    _, Y = mwc_mean_bound(PO2, n=n, KR=KR, KT=KT, L0=L0_eff)
    plt.plot(PO2, Y, label=sc["label"])

plt.xscale('log')
plt.xlabel('PO$_2$ [mmHg]')
plt.ylabel('Saturation (Y)')
plt.yticks([0,0.25,0.5,0.75,1.0])
plt.title('MWC Oxygen Dissociation Curves with Bohr Shift')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.show()

# ===== 参考：P50（Y=0.5 となるPO2）を数値で求めて表示 =====
def p50(PO2_grid, Y):
    # 単純補間でY=0.5に最も近い点
    idx = np.argmin(np.abs(Y - 0.5))
    return PO2_grid[idx]

print("\n--- P50 (近似) ---")
for sc in scenarios:
    L0_eff = L0_with_bohr(sc["PCO2"], sc["pH"], L0_base=L0_base)
    _, Y = mwc_mean_bound(PO2, n=n, KR=KR, KT=KT, L0=L0_eff)
    print(f'{sc["label"]}: P50 ≈ {p50(PO2, Y):.1f} mmHg')
