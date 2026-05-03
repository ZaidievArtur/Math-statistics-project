import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_laguerre
from scipy.integrate import quad

# ==================== ПАРАМЕТРЫ ====================
np.random.seed(42)

n = 400
lam = 1
c = 2
MAX_N = 50

# ==================== ГЕНЕРАЦИЯ ВЫБОРКИ ====================
# Метод обратной функции для Weibull
u = np.random.uniform(0, 1, n)
sample = (-np.log(u) / lam) ** (1 / c)

# ==================== ИСТИННАЯ ПЛОТНОСТЬ ====================
def f(x):
    return 2 * x * np.exp(-x**2)

# ==================== ФУНКЦИИ ЛАГЕРРА ====================
def phi(i, x):
    return np.exp(-x / 2) * eval_laguerre(i, x)

# ==================== КОЭФФИЦИЕНТЫ ====================
true_c = np.zeros(MAX_N + 1)
hat_c = np.zeros(MAX_N + 1)

print("Вычисляем коэффициенты...")

for i in range(MAX_N + 1):
    true_c[i], _ = quad(lambda x: f(x) * phi(i, x), 0, np.inf)
    hat_c[i] = np.mean(phi(i, sample))

# ==================== ТАБЛИЦА ====================
print("\nКоэффициенты:")
print("-" * 45)
print(" i | true_c_i      | hat_c_i")
print("-" * 45)

for i in range(MAX_N + 1):
    print(f"{i:2d} | {true_c[i]:.8f} | {hat_c[i]:.8f}")

# ==================== ОЦЕНКА ОШИБКИ ====================
N_values = list(range(5, 51, 5))
delta_n = []

for N in N_values:
    disp = np.sum((hat_c[:N+1] - true_c[:N+1])**2)
    tail = np.sum(true_c[N+1:]**2)
    delta_n.append(disp + tail)

delta_n = np.array(delta_n)

best_N = N_values[np.argmin(delta_n)]

# ==================== ГРАФИКИ ПЛОТНОСТЕЙ ====================
x = np.linspace(0, 3, 1000)
f_true = f(x)

plt.figure(figsize=(14, 10))

for idx, N in enumerate([5, 15, 25, 35, 50], 1):
    plt.subplot(3, 2, idx)

    f_hat = np.zeros_like(x)

    for i in range(N + 1):
        f_hat += hat_c[i] * phi(i, x)

    plt.plot(x, f_true, label='Истинная плотность')
    plt.plot(x, f_hat, '--', label=f'N={N}')

    plt.title(f'Проекционная оценка, N={N}')
    plt.grid(True)
    plt.legend()

# ==================== ГРАФИК delta_n ====================
plt.subplot(3, 2, 6)
plt.plot(N_values, delta_n, 'o-')
plt.title('Зависимость δₙ(N)')
plt.xlabel('N')
plt.ylabel('δₙ')
plt.grid(True)

plt.tight_layout()
plt.show()

# ==================== ИТОГ ====================
print("\n" + "="*50)
print(f"Минимальная ошибка при N = {best_N}")
print(f"δ_n = {np.min(delta_n):.8f}")
print("="*50)