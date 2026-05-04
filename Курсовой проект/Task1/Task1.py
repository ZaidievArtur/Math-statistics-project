import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import erfc
import os

# ================================
# ПАРАМЕТР РАСПРЕДЕЛЕНИЯ
# ================================
lambda_val = 1.0

# ================================
# СОЗДАНИЕ ПАПКИ
# ================================
os.makedirs("figures", exist_ok=True)

# ================================
# АНАЛИТИЧЕСКИЕ ИНТЕГРАЛЫ
# ================================

def I1_const():
    return np.sqrt(np.pi)

def I4_const():
    return np.pi * lambda_val

def I2_func(xi):
    xi = np.maximum(xi, 1e-12)
    return np.pi * lambda_val * np.exp((lambda_val * xi)**2) * erfc(lambda_val * xi)

def I3_func(xi):
    xi = np.maximum(xi, 1e-12)
    return np.pi * lambda_val * np.exp((lambda_val * xi)**2 / 2) * erfc(lambda_val * xi / np.sqrt(2))

# ================================
# ОИСКО
# ================================

def calculate_mise(xi, n):
    term1 = 1.0
    term2 = I1_const() / (n * xi * I4_const())
    term3 = (1 - 1/n) * I2_func(xi) / I4_const()
    term4 = -2 * I3_func(xi) / I4_const()
    return term1 + term2 + term3 + term4

# ================================
# ПОИСК ОПТИМУМА
# ================================

def optimize_xi(n):
    res = minimize_scalar(
        lambda x: calculate_mise(x, n),
        bounds=(0.01, 5.0),
        method='bounded'
    )
    return res.x, res.fun

# ================================
# НАСТРОЙКИ
# ================================

n_values = [10, 50, 250, 1000]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
xi_grid = np.linspace(0.01, 3.0, 1000)

# Диапазон n
n_range = np.logspace(1, 4, 100).astype(int)

opt_xis = []
min_mises = []

for n in n_range:
    ox, mm = optimize_xi(n)
    opt_xis.append(ox)
    min_mises.append(mm)

# Предвычисление
opt_dict = {}
min_dict = {}

for n in n_values:
    ox, mm = optimize_xi(n)
    opt_dict[n] = ox
    min_dict[n] = mm

# ================================
# ГРАФИК 1
# ОИСКО
# ================================

fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111)

for n, color in zip(n_values, colors):
    vals = calculate_mise(xi_grid, n)
    ox = opt_dict[n]
    mm = min_dict[n]

    label = f'n={n}, ξ*={ox:.3f}, δ̄min={mm:.4f}'

    ax1.plot(xi_grid, vals, color=color, lw=2.5, label=label)
    ax1.plot(ox, mm, 'o', color=color, markersize=8)

ax1.set_title('1. ОИСКО δ̄ₙ(ξ)')
ax1.set_xlabel('ξ')
ax1.set_ylabel('δ̄ₙ')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

plt.tight_layout()
plt.savefig("figures/f1.png", dpi=300)

# ================================
# ГРАФИК 2
# ξopt(n)
# ================================

fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)

ax2.plot(n_range, opt_xis, color='tab:purple', lw=2.5)

ax2.set_xscale('log')
ax2.set_title('2. Зависимость ξopt от n')
ax2.set_xlabel('n')
ax2.set_ylabel('ξopt')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/f2.png", dpi=300)

# ================================
# ГРАФИК 3
# δmin(n)
# ================================

fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111)

ax3.plot(n_range, min_mises, color='tab:brown', lw=2.5)

ax3.set_xscale('log')
ax3.set_title('3. Нижняя граница δ̄min')
ax3.set_xlabel('n')
ax3.set_ylabel('δ̄min')
ax3.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/f3.png", dpi=300)

# ================================
# ГРАФИК 4
# ЭФФЕКТИВНОСТЬ
# ================================

fig4 = plt.figure(figsize=(10, 6))
ax4 = fig4.add_subplot(111)

for n, color in zip(n_values, colors):
    vals = calculate_mise(xi_grid, n)
    mm = min_dict[n]
    ox = opt_dict[n]

    eff = mm / vals

    ax4.plot(xi_grid, eff, color=color, lw=2.5,
             label=f'n={n}, ξ*={ox:.3f}')
    ax4.plot(ox, 1.0, 'o', color=color, markersize=8)

ax4.set_title('4. Эффективность eₙ(ξ)')
ax4.set_xlabel('ξ')
ax4.set_ylabel('eₙ(ξ)')
ax4.set_ylim(0, 1.1)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend()

plt.tight_layout()
plt.savefig("figures/f4.png", dpi=300)

# ================================
# ПОКАЗАТЬ ВСЁ
# ================================

plt.show()