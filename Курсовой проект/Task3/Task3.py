import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import erfc
import os

os.makedirs("figures", exist_ok=True)

LAMBDA = 1.0

# ====================== 1. ОМП ======================
def calculate_mise_omp(n):
    if n <= 2:
        return np.inf
    return (2.0 * n) / ((n - 1) * (n - 2))

# ====================== 2. ЯДЕРНАЯ ОЦЕНКА ======================
def I1_const():
    return np.sqrt(np.pi)

def I4_const():
    return np.pi * LAMBDA

def I2_func(xi):
    xi = np.maximum(xi, 1e-12)
    return np.pi * LAMBDA * np.exp((LAMBDA * xi)**2) * erfc(LAMBDA * xi)

def I3_func(xi):
    xi = np.maximum(xi, 1e-12)
    return np.pi * LAMBDA * np.exp((LAMBDA * xi)**2 / 4) * erfc(LAMBDA * xi / np.sqrt(2))

def calculate_mise_kernel(xi, n):
    term1 = 1.0
    term2 = I1_const() / (n * xi * I4_const())
    term3 = (1 - 1.0 / n) * I2_func(xi) / I4_const()
    term4 = -2.0 * I3_func(xi) / I4_const()
    return term1 + term2 + term3 + term4

def optimize_kernel(n):
    res = minimize_scalar(
        lambda x: calculate_mise_kernel(x, n),
        bounds=(0.01, 3.0),
        method='bounded',
        options={'xatol': 1e-9, 'maxiter': 1000}
    )
    return res.x, res.fun

# ====================== 3. ГИСТОГРАММА ======================
def calculate_mise_hist(xi, n):
    xi = max(xi, 1e-9)
    var_term = 1.0 / (n * xi)
    bias_term = (xi**2) / 12
    return 1.0 + var_term + bias_term

def optimize_hist(n):
    res = minimize_scalar(
        lambda x: calculate_mise_hist(x, n),
        bounds=(0.05, 5.0),
        method='bounded',
        options={'xatol': 1e-8}
    )
    return res.x, res.fun

# ====================== ПАРАМЕТРЫ ======================
n_range = np.unique(np.logspace(1, 5.5, 200).astype(int))  # расширили диапазон

# Предвычисления
omp_vals = np.array([calculate_mise_omp(n) for n in n_range])
kernel_min = []
hist_min = []

print("Идёт предвычисление...")
for n in n_range:
    _, mm_k = optimize_kernel(n)
    _, mm_h = optimize_hist(n)
    kernel_min.append(mm_k)
    hist_min.append(mm_h)

kernel_min = np.array(kernel_min)
hist_min = np.array(hist_min)

# ====================== ФУНКЦИЯ ПОИСКА ПЕРЕСЕЧЕНИЯ ======================
def find_crossing(n_arr, y1, y2, label=""):
    diff = y1 - y2
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) > 0:
        i = idx[0]
        n1, n2 = n_arr[i], n_arr[i+1]
        d1, d2 = diff[i], diff[i+1]
        n_cr = n1 - d1 * (n2 - n1) / (d2 - d1)
        print(f"Найдено пересечение {label}: n_cr ≈ {n_cr:.1f}")
        return n_cr
    else:
        print(f"Пересечение {label} не найдено в диапазоне")
        return None

n_cr_kernel = find_crossing(n_range, kernel_min, omp_vals, "Ядерная vs ОМП")
n_cr_hist   = find_crossing(n_range, hist_min, omp_vals, "Гистограмма vs ОМП")

# ====================== ГРАФИК 1 ======================
fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.add_subplot(111)
ax1.plot(n_range, omp_vals, color='tab:brown', lw=3, label='ОМП')
ax1.set_title('1. Нижняя граница $\\bar{\\delta}_{n,\\min}$ (ОМП)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Объём выборки $n$')
ax1.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which="both", linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
plt.savefig("figures/f9.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 2: Ядерная vs ОМП ======================
fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(n_range, kernel_min, 'b-', lw=2.5, label='Ядерная оценка')
ax2.plot(n_range, omp_vals, 'r-', lw=3, label='ОМП')

if n_cr_kernel:
    ax2.axvline(x=n_cr_kernel, color='k', linestyle='--', alpha=0.8)
    ax2.plot(n_cr_kernel, calculate_mise_omp(int(n_cr_kernel)), 'ko', markersize=8)
    ax2.annotate(f'$n_{{кр}} \\approx {int(n_cr_kernel)}$',
                 xy=(n_cr_kernel, 0.1), xytext=(n_cr_kernel*1.8, 0.18),
                 arrowprops=dict(arrowstyle="->"), fontsize=12)

ax2.set_title('2. Сравнение: Ядерная оценка vs ОМП', fontsize=14, fontweight='bold')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which="both", linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
plt.savefig("figures/f10.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 3: Гистограмма vs ОМП ======================
fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(n_range, hist_min, 'g-', lw=2.5, label='Гистограмма')
ax3.plot(n_range, omp_vals, 'r-', lw=3, label='ОМП')

if n_cr_hist:
    ax3.axvline(x=n_cr_hist, color='k', linestyle='--', alpha=0.8)
    ax3.plot(n_cr_hist, calculate_mise_omp(int(n_cr_hist)), 'ko', markersize=8)
    ax3.annotate(f'$n_{{кр}} \\approx {int(n_cr_hist)}$',
                 xy=(n_cr_hist, 0.1), xytext=(n_cr_hist*1.8, 0.18),
                 arrowprops=dict(arrowstyle="->"), fontsize=12)

ax3.set_title('3. Сравнение: Гистограмма vs ОМП', fontsize=14, fontweight='bold')
ax3.set_xlabel('$n$')
ax3.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(True, which="both", linestyle='--', alpha=0.7)
ax3.legend()
plt.tight_layout()
plt.savefig("figures/f11.png", dpi=300, bbox_inches='tight')

plt.show()

print("\nГотово! Графики сохранены.")