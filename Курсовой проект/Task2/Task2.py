import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import erfc
import os

os.makedirs("figures", exist_ok=True)

LAMBDA = 1.0


# ==========================================
# ЯДЕРНАЯ ОЦЕНКА (колокольное ядро)
# ==========================================

def I1_const():
    return np.sqrt(np.pi)


def I4_const():
    return np.pi * LAMBDA


def I2_func(xi):
    xi = np.maximum(xi, 1e-12)
    return np.pi * LAMBDA * np.exp((LAMBDA * xi) ** 2) * erfc(LAMBDA * xi)


def I3_func(xi):
    xi = np.maximum(xi, 1e-12)
    return np.pi * LAMBDA * np.exp((LAMBDA * xi) ** 2 / 4) * erfc(LAMBDA * xi / np.sqrt(2))


def calculate_delta_kernel(xi, n):
    term1 = 1.0
    term2 = I1_const() / (n * xi * I4_const())
    term3 = (1 - 1 / n) * I2_func(xi) / I4_const()
    term4 = -2 * I3_func(xi) / I4_const()
    return term1 + term2 + term3 + term4


def optimize_kernel(n):
    res = minimize_scalar(
        lambda x: calculate_delta_kernel(x, n),
        bounds=(0.01, 2.0),
        method='bounded',
        options={'xatol': 1e-8}
    )
    return res.x, res.fun


# ==========================================
# ГИСТОГРАММА (для сравнения)
# ==========================================

def calculate_delta_hist(xi, n):
    """Примерная реализация для гистограммы (можно уточнить по методичке)"""
    xi = max(xi, 1e-9)
    # Для показательного распределения на [0, ∞) — упрощённая формула
    term_var = 1.0 / (n * xi)
    # Пример bias-терма (можно улучшить)
    bias_term = (xi ** 2 * LAMBDA ** 2) / 12
    return 1.0 + term_var + bias_term


def optimize_hist(n):
    res = minimize_scalar(
        lambda x: calculate_delta_hist(x, n),
        bounds=(0.01, 3.0),
        method='bounded',
        options={'xatol': 1e-8}
    )
    return res.x, res.fun


# ==========================================
# ПАРАМЕТРЫ
# ==========================================

n_values = [10, 50, 250, 1000]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']

xi_grid = np.linspace(0.01, 1.5, 1200)
n_range = np.unique(np.logspace(1, 4, 150).astype(int))

# Предвычисления
opt_kernel_dict = {}
min_kernel_dict = {}
opt_hist_dict = {}
min_hist_dict = {}

for n in n_values:
    ox_k, mm_k = optimize_kernel(n)
    ox_h, mm_h = optimize_hist(n)
    opt_kernel_dict[n] = ox_k
    min_kernel_dict[n] = mm_k
    opt_hist_dict[n] = ox_h
    min_hist_dict[n] = mm_h

# Для графиков 2 и 3 (ядерная оценка)
opt_xis_kernel = []
min_deltas_kernel = []
opt_xis_hist = []
min_deltas_hist = []

for n in n_range:
    ox_k, mm_k = optimize_kernel(n)
    ox_h, mm_h = optimize_hist(n)
    opt_xis_kernel.append(ox_k)
    min_deltas_kernel.append(mm_k)
    opt_xis_hist.append(ox_h)
    min_deltas_hist.append(mm_h)

# ==========================================
# ГРАФИК 1 — Ядерная оценка
# ==========================================

fig1 = plt.figure(figsize=(10, 6.5))
ax1 = fig1.add_subplot(111)

for n, color in zip(n_values, colors):
    vals = calculate_delta_kernel(xi_grid, n)
    ox = opt_kernel_dict[n]
    mm = min_kernel_dict[n]
    ax1.plot(xi_grid, vals, color=color, lw=2.5,
             label=f'$n={n}$, $\\xi^*={ox:.3f}$, $\\bar\\delta={mm:.4f}$')
    ax1.plot(ox, mm, 'o', color=color, markeredgecolor='black', markersize=9, zorder=10)

ax1.set_title('1. ОИСКО ядерной оценки $\\bar{\\delta}_n(\\xi)$', fontsize=14, fontweight='bold')
ax1.set_xlabel('$\\xi$')
ax1.set_ylabel('$\\bar{\\delta}_n(\\xi)$')
ax1.set_xlim(0, 0.8)
ax1.set_ylim(0, 0.4)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)
plt.tight_layout()
plt.savefig("figures/f1_kernel.png", dpi=300, bbox_inches='tight')

# ==========================================
# ГРАФИК 2 — ξ_opt(n)
# ==========================================

fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(n_range, opt_xis_kernel, color='tab:purple', lw=2.6, label='Ядерная оценка')
ax2.plot(n_range, opt_xis_hist, color='tab:gray', lw=2.6, linestyle='--', label='Гистограмма')
ax2.set_xscale('log')
ax2.set_title('2. Оптимальный параметр $\\xi_{opt}$', fontsize=14, fontweight='bold')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$\\xi_{opt}$')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
plt.savefig("figures/f2_xi_opt.png", dpi=300, bbox_inches='tight')

# ==========================================
# ГРАФИК 3 — Нижняя граница
# ==========================================

fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(n_range, min_deltas_kernel, color='tab:brown', lw=2.6, label='Ядерная оценка')
ax3.plot(n_range, min_deltas_hist, color='tab:gray', lw=2.6, linestyle='--', label='Гистограмма')
ax3.set_xscale('log')
ax3.set_title('3. Нижняя граница погрешности $\\bar{\\delta}_{n,min}$', fontsize=14, fontweight='bold')
ax3.set_xlabel('$n$')
ax3.set_ylabel('$\\bar{\\delta}_{n,min}$')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
plt.tight_layout()
plt.savefig("figures/f3_delta_min.png", dpi=300, bbox_inches='tight')

# ==========================================
# ГРАФИК 4 — СРАВНЕНИЕ ЯДЕРНАЯ vs ГИСТОГРАММА
# ==========================================

fig4 = plt.figure(figsize=(10, 6.5))
ax4 = fig4.add_subplot(111)

ax4.plot(n_range, min_deltas_kernel, 'b-', lw=2.7, label='Ядерная оценка (колокольное ядро)')
ax4.plot(n_range, min_deltas_hist, 'r-', lw=2.7, label='Гистограмма')

# Поиск точки пересечения
diff = np.array(min_deltas_kernel) - np.array(min_deltas_hist)
cross_idx = np.where(np.diff(np.sign(diff)))[0]

if len(cross_idx) > 0:
    idx = cross_idx[0]
    n1, n2 = n_range[idx], n_range[idx + 1]
    d1, d2 = diff[idx], diff[idx + 1]
    n_cr = n1 - d1 * (n2 - n1) / (d2 - d1)
    y_cr = min_deltas_kernel[idx] + (n_cr - n1) * (min_deltas_kernel[idx + 1] - min_deltas_kernel[idx]) / (n2 - n1)

    ax4.axvline(x=n_cr, color='k', linestyle='--', alpha=0.8)
    ax4.plot(n_cr, y_cr, 'ko', markersize=8)
    ax4.annotate(f'$n_{{кр}} \\approx {int(n_cr)}$',
                 xy=(n_cr, y_cr), xytext=(n_cr * 1.8, y_cr + 0.01),
                 arrowprops=dict(arrowstyle="->"), fontsize=12)

ax4.set_title('4. Сравнение ОИСКО: ядерная оценка vs гистограмма', fontsize=14, fontweight='bold')
ax4.set_xlabel('Объём выборки $n$')
ax4.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax4.set_xscale('log')
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend(fontsize=11)
plt.tight_layout()
plt.savefig("figures/f4_comparison.png", dpi=300, bbox_inches='tight')

plt.show()

# ==========================================
print("✅ Все графики сохранены!")
print("\nОптимальные параметры (ядерная оценка):")
for n in n_values:
    print(f"n={n:4d} | ξ*={opt_kernel_dict[n]:.4f} | δ_min={min_kernel_dict[n]:.5f}")