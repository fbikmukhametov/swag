import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jvp, yvp, hankel1

# ==========================================
# 1. Параметры
# ==========================================
R         = 0.004
kR        = 5.1291
k         = kR / R
theta_deg = 10
theta     = np.deg2rad(theta_deg)
M         = 20
m_vals    = np.arange(-M, M + 1)
n         = len(m_vals)

d  = 0.01
r1 = np.array([-d/2, 0.0])
r2 = np.array([ d/2, 0.0])

theta0_1 = 0.0
theta0_2 = np.pi
nb_inc   = 5

# ==========================================
# 2. Вспомогательные матрицы
# ==========================================
Jp    = np.diag(jvp(m_vals, kR))
Hp    = np.diag(jvp(m_vals, kR) + 1j * yvp(m_vals, kR))
J_m_R = np.diag(jv(m_vals, kR))
H_m_R = np.diag(jv(m_vals, kR) + 1j * yv(m_vals, kR))
I_n   = np.eye(n, dtype=complex)

# ==========================================
# 3. T-матрица
# ==========================================
def compute_T(theta0):
    PA = np.zeros((n, n), dtype=complex)
    for i, m in enumerate(m_vals):
        for j, mp in enumerate(m_vals):
            if i == j:
                PA[i, j] = theta / np.pi
            else:
                PA[i, j] = (np.sin((m - mp) * theta) / (np.pi * (m - mp))) \
                           * np.exp(1j * (m - mp) * theta0)
    Hp_inv   = np.diag(1.0 / np.diag(Hp))
    HpJp_inv = np.diag(1.0 / np.diag(Hp @ Jp))
    T_HW  = -Hp_inv @ Jp
    K_mat = PA @ HpJp_inv @ PA
    W_out = Hp_inv @ PA
    W_in  = PA @ Hp_inv
    lam = 3e-4 * np.linalg.norm(K_mat)
    X = np.linalg.solve(K_mat + lam * np.eye(n, dtype=complex), W_in)
    return T_HW + W_out @ X, PA

T1, PA1 = compute_T(theta0_1)
T2, PA2 = compute_T(theta0_2)

# ==========================================
# 4. Матрицы перехода
# ==========================================
def compute_p(d_vec):
    dist  = np.linalg.norm(d_vec)
    angle = np.arctan2(d_vec[1], d_vec[0])
    P = np.zeros((n, n), dtype=complex)
    for i, q in enumerate(m_vals):
        for j, l in enumerate(m_vals):
            P[i, j] = hankel1(l - q, k * dist) * np.exp(1j * (l - q) * angle)
    return P

p_12 = compute_p(r1 - r2)
p_21 = compute_p(r2 - r1)

# ==========================================
# 5. Матрица X и правая часть
# ==========================================
X = np.zeros((2*n, 2*n), dtype=complex)
X[:n, :n] = I_n
X[:n, n:] = -T1 @ p_12
X[n:, :n] = -T2 @ p_21
X[n:, n:] = I_n

a1 = np.array([jv(nb_inc - m, k * d/2) * np.exp(1j * (nb_inc - m) * np.pi) for m in m_vals])
a2 = np.array([jv(nb_inc - m, k * d/2) for m in m_vals])

rhs = np.concatenate([T1 @ a1, T2 @ a2])
b   = np.linalg.solve(X, rhs)
b1  = b[:n]
b2  = b[n:]

# ==========================================
# 6. Внутренние коэффициенты (с учётом локального поля от соседа)
# ==========================================
def compute_c(a_vec, b_vec, PA):
    M_int = PA @ J_m_R + (I_n - PA) @ Jp
    rhs_c = PA @ (J_m_R @ a_vec + H_m_R @ b_vec)
    return np.linalg.solve(M_int, rhs_c)

a1_local = a1 + p_12 @ b2
a2_local = a2 + p_21 @ b1

c1 = compute_c(a1_local, b1, PA1)
c2 = compute_c(a2_local, b2, PA2)

# ==========================================
# 7. Сетка
# ==========================================
N_grid = 200
x = np.linspace(-0.02, 0.02, N_grid)
y = np.linspace(-0.02, 0.02, N_grid)
Xg, Yg = np.meshgrid(x, y)

# Падающее поле
r_global   = np.sqrt(Xg**2 + Yg**2)
phi_global = np.arctan2(Yg, Xg)
p_inc = jv(nb_inc, k * r_global) * np.exp(1j * nb_inc * phi_global)

# Локальные вклады каждого цилиндра
def local_field(rj, coeffs, kind):
    dx = Xg - rj[0]; dy = Yg - rj[1]
    r_loc = np.sqrt(dx**2 + dy**2)
    phi_loc = np.arctan2(dy, dx)
    p_loc = np.zeros_like(Xg, dtype=complex)
    for i_m, m in enumerate(m_vals):
        if kind == 'sca':
            p_loc += coeffs[i_m] * hankel1(m, k*r_loc) * np.exp(1j*m*phi_loc)
        else:
            p_loc += coeffs[i_m] * jv(m, k*r_loc) * np.exp(1j*m*phi_loc)
    return p_loc, r_loc

p_sca_1, r_1 = local_field(r1, b1, 'sca')
p_sca_2, r_2 = local_field(r2, b2, 'sca')
p_int_1, _   = local_field(r1, c1, 'int')
p_int_2, _   = local_field(r2, c2, 'int')

mask_in_1 = r_1 <= R
mask_in_2 = r_2 <= R
mask_out  = ~mask_in_1 & ~mask_in_2

# === СБОРКА ПОЛЯ БЕЗ ДВОЙНОГО СЧЁТА ===
p_total = p_inc.copy()
p_total[mask_out]  += (p_sca_1[mask_out] + p_sca_2[mask_out])
p_total[mask_in_1]  = p_int_1[mask_in_1]     # внутри цилиндра 1 — только внутреннее поле
p_total[mask_in_2]  = p_int_2[mask_in_2]     # внутри цилиндра 2 — только внутреннее поле

p_sca_total = p_total - p_inc
p_plot = np.real(p_sca_total)

# ==========================================
# 8. График
# ==========================================
fig, ax = plt.subplots(figsize=(8, 7))
vmax = np.abs(p_plot).max()
im = ax.pcolormesh(Xg, Yg, p_plot, cmap='bwr',
                   vmin=-vmax, vmax=vmax, shading='auto')

# Правильная отрисовка вырезов
for rj, theta0 in zip([r1, r2], [theta0_1, theta0_2]):
    phi_arc = np.linspace(theta0 + theta, theta0 + 2*np.pi - theta, 500)
    ax.plot(rj[0] + R*np.cos(phi_arc), rj[1] + R*np.sin(phi_arc),
            'k-', linewidth=2.5)

ax.set_aspect('equal')
ax.set_xlim(-0.02, 0.02)
ax.set_ylim(-0.02, 0.02)
ax.set_xlabel('$x$ (m)', fontsize=12)
ax.set_ylabel('$y$ (m)', fontsize=12)
ax.set_title(f'Scattered field, $n_b={nb_inc}$, $kR={kR}$', fontsize=13)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\mathrm{Re}(p_t - p_{inc})$', fontsize=12)
plt.tight_layout()

print("Max |p_sca| (Python) =", np.abs(p_plot).max())
print("Min |p_sca| (Python) =", np.abs(p_plot).min())
print("Value in (0, 0.012):", np.real(jv(5, k*np.sqrt(0.012**2)) * np.exp(1j*5*np.arctan2(0.012, 0))) )
plt.show()

# Значения поля в двух ключевых точках
i_R  = np.argmin(np.abs(x - 0.008))   # точка справа между центрами
i_L  = np.argmin(np.abs(x + 0.008))   # точка слева
j_0  = np.argmin(np.abs(y))

print("\n=== Точки сравнения ===")
print(f"p_sca(+0.008, 0) = {p_plot[j_0, i_R]:+.4f}")
print(f"p_sca(-0.008, 0) = {p_plot[j_0, i_L]:+.4f}")
print(f"Сумма = {p_plot[j_0, i_R] + p_plot[j_0, i_L]:+.4f}  (если ≈ 0 — антисимметрия есть)")