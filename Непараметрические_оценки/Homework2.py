import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========================= НАСТРОЙКИ =========================
n = 400
lam = 0.5
c = 2
np.random.seed(42)

# ========================= ПЛОТНОСТЬ =========================
def pdf(x):
    return lam * c * x**(c - 1) * np.exp(-lam * x**c)

# Норма ||f||²
Norm_L2_sq, _ = quad(lambda x: pdf(x)**2, 0, np.inf)

# ====================== ГЕНЕРАЦИЯ ВЫБОРКИ ======================
def generate_sample(size):
    u = np.random.uniform(0, 1, size)
    return (-np.log(u) / lam) ** (1 / c)

sample_single = generate_sample(n)

x_min = np.min(sample_single)
x_max = np.max(sample_single)
Delta = x_max - x_min

print(f"Выборка сгенерирована (n={n}):")
print(f"x_min = {x_min:.6f}, x_max = {x_max:.6f}, Δ = {Delta:.6f}\n")

# ====================== ВЫЧИСЛЕНИЕ ОИСКО ======================
def compute_delta_n(m, sample, x_min, x_max):
    if m < 1:
        return np.nan

    h = (x_max - x_min) / m
    bins = np.linspace(x_min, x_max, m + 1)
    counts, _ = np.histogram(sample, bins=bins)

    ise = 0.0

    for i in range(m):
        left, right = bins[i], bins[i + 1]
        f_hat = counts[i] / (n * h)

        integrand = lambda x: (f_hat - pdf(x))**2
        integral, _ = quad(integrand, left, right)

        ise += integral

    # хвосты
    if x_min > 0:
        left_tail, _ = quad(lambda x: pdf(x)**2, 0, x_min)
        ise += left_tail

    right_tail, _ = quad(lambda x: pdf(x)**2, x_max, np.inf)
    ise += right_tail

    return ise / Norm_L2_sq

# ====================== ГИСТОГРАММЫ ======================
fig1, axs1 = plt.subplots(2, 2, figsize=(16, 10))

m_hist = [10, 20, 30, 40]

for idx, m in enumerate(m_hist):
    h = Delta / m
    bins = np.linspace(x_min, x_max, m + 1)
    counts, _ = np.histogram(sample_single, bins=bins)

    bin_centers = x_min + (np.arange(m) + 0.5) * h
    empirical_density = counts / (n * h)

    ax = axs1[idx // 2, idx % 2]

    ax.bar(
        bin_centers,
        empirical_density,
        width=h,
        alpha=0.7,
        edgecolor='black',
        label='Гистограмма'
    )

    x_plot = np.linspace(0, x_max + 1, 1000)
    ax.plot(x_plot, pdf(x_plot), 'r-', lw=2.5, label='Теоретическая f(x)')

    delta = compute_delta_n(m, sample_single, x_min, x_max)

    ax.set_title(f'm={m}, δₙ={delta:.5f}')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# ====================== ВЫВОД ПО ГИСТОГРАММАМ ======================
for m in m_hist:
    delta = compute_delta_n(m, sample_single, x_min, x_max)
    empty = np.sum(
        np.histogram(sample_single, bins=np.linspace(x_min, x_max, m + 1))[0] == 0
    )

    print(f"m = {m:2d} | h = {Delta/m:.5f} | пустых разрядов = {empty:2d} | δ_n = {delta:.6f}")

# ====================== ЗАВИСИМОСТЬ ОШИБКИ ОТ m ======================
m_all = list(range(1, 41))
delta_values = []

for m in m_all:
    delta_values.append(
        compute_delta_n(m, sample_single, x_min, x_max)
    )

plt.figure(figsize=(10, 6))
plt.plot(m_all, delta_values, 'o-', lw=2)

plt.title('Зависимость ОИСКО от числа разрядов m')
plt.xlabel('Число разрядов m')
plt.ylabel('δₙ(m)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ====================== ТАБЛИЦА ======================
print("\n" + "="*55)
print(" m | h       | пустых разрядов | ОИСКО δ_n")
print("-"*55)

for m in m_all:
    h = Delta / m
    counts = np.histogram(sample_single, bins=np.linspace(x_min, x_max, m+1))[0]
    empty = np.sum(counts == 0)
    delta = compute_delta_n(m, sample_single, x_min, x_max)

    print(f"{m:2d} | {h:.5f} | {empty:2d}              | {delta:.6f}")

print("="*55)

# ====================== ИТОГ ======================
min_m = m_all[np.argmin(delta_values)]

print(f"\nМинимум ОИСКО достигается при m = {min_m}")
print(f"δ_n ≈ {min(delta_values):.6f}")