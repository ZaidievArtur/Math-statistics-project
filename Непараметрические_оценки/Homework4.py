import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ===================== ПАРАМЕТРЫ =====================
n = 400
lam = 1
c = 2
np.random.seed(42)

# ===================== ГЕНЕРАЦИЯ ВЫБОРКИ =====================
# Метод обратной функции для распределения Вейбулла
u = np.random.uniform(0, 1, n)
sample = (-np.log(u) / lam) ** (1 / c)

# ===================== ТЕОРЕТИЧЕСКАЯ ПЛОТНОСТЬ =====================
def f(x):
    return lam * c * x**(c - 1) * np.exp(-lam * x**c)

# ===================== ГАУССОВО ЯДРО =====================
def K(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-u**2 / 2)

# ===================== ЯДЕРНАЯ ОЦЕНКА =====================
def f_hat(x, h, sample):
    x = np.atleast_1d(x)
    return np.sum(K((x[:, None] - sample) / h), axis=1) / (n * h)

# ===================== ОИСКО =====================
# Относительная интегральная среднеквадратичная ошибка
norm_L2, _ = quad(lambda x: f(x)**2, 0, np.inf)

h_values = np.arange(0.05, 1.51, 0.02)
delta_values = []

print("Вычисляем ОИСКО...")

for h in h_values:
    def integrand(x):
        return (f_hat(np.array([x]), h, sample)[0] - f(x))**2

    ise, _ = quad(integrand, 0, 5, limit=300)
    delta_values.append(ise / norm_L2)

delta_values = np.array(delta_values)

best_idx = np.argmin(delta_values)
best_h = h_values[best_idx]

# ===================== ТАБЛИЦА =====================
print("\n" + "="*50)
print("   h      |   ОИСКО")
print("-"*50)

for i in range(0, len(h_values), 5):
    print(f"{h_values[i]:.3f}   |   {delta_values[i]:.6f}")

print("="*50)
print(f"Минимум: h = {best_h:.3f}, ОИСКО = {delta_values[best_idx]:.6f}")

# ===================== ГРАФИКИ =====================
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# График ОИСКО(h)
axs[0].plot(h_values, delta_values, 'o-', lw=2)
axs[0].plot(best_h, delta_values[best_idx], 'ro', markersize=10)
axs[0].set_title("Зависимость ОИСКО от h")
axs[0].set_xlabel("h")
axs[0].set_ylabel("ОИСКО")
axs[0].grid(True)

# Сравнение плотностей
x_plot = np.linspace(0, 3, 1000)
y_true = f(x_plot)
y_hat = f_hat(x_plot, best_h, sample)

axs[1].plot(x_plot, y_true, 'r', lw=2.5, label='Теоретическая f(x)')
axs[1].plot(x_plot, y_hat, 'b--', lw=2.5,
            label=f'Ядерная оценка (h={best_h:.3f})')

axs[1].hist(sample, bins=25, density=True, alpha=0.5,
            edgecolor='black', label='Гистограмма')

axs[1].set_title("Ядерная оценка плотности")
axs[1].set_xlabel("x")
axs[1].set_ylabel("Плотность")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()