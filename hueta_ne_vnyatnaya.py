import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jvp, yvp, hankel1

# ==========================================
# 1. Параметры
# ==========================================
R         = 0.04
kR        = 5.1291
k         = kR / R
theta_deg = 10
theta     = np.deg2rad(theta_deg)
theta0    = 0
nb_inc    = 1
M         = 20
m_vals    = np.arange(-M, M + 1)
n         = len(m_vals)

# ==========================================
# 2. T-матрица (формула 6) с регуляризацией Тихонова
# ==========================================
Jp = np.diag(jvp(m_vals, kR))
Hp = np.diag(jvp(m_vals, kR) + 1j * yvp(m_vals, kR))
J_m_R = np.diag(jv(m_vals, kR))
H_m_R = np.diag(jv(m_vals, kR) + 1j * yv(m_vals, kR))

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
T = T_HW + W_out @ X

# ==========================================
# 3. Векторы a и f
# ==========================================
a_vec = np.zeros(n, dtype=complex)
a_vec[list(m_vals).index(nb_inc)] = 1.0
f_vec = T @ a_vec

# ==========================================
# 4. Внутренние коэффициенты c_m
#    [P_A J + (I - P_A) J'] c = P_A (J a + H f)
# ==========================================
I_n   = np.eye(n, dtype=complex)
M_int = PA @ J_m_R + (I_n - PA) @ Jp
rhs   = PA @ (J_m_R @ a_vec + H_m_R @ f_vec)
c_vec = np.linalg.solve(M_int, rhs)

# ==========================================
# 5. Сетка (x, y)
# ==========================================
N_grid = 600
x = np.linspace(-0.15, 0.15, N_grid)
y = np.linspace(-0.15, 0.15, N_grid)
X, Y = np.meshgrid(x, y)

r   = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

# ==========================================
# 6. p_bg (падающее поле) — определено ВЕЗДЕ
# ==========================================
p_bg = np.zeros_like(X, dtype=complex)
for i_m, m in enumerate(m_vals):
    p_bg += a_vec[i_m] * jv(m, k * r) * np.exp(1j * m * phi)

# ==========================================
# 7. p_int (внутри) и p_ext (снаружи)
# ==========================================
p_int = np.zeros_like(X, dtype=complex)
p_ext = np.zeros_like(X, dtype=complex)
for i_m, m in enumerate(m_vals):
    phase = np.exp(1j * m * phi)
    Jm = jv(m, k * r)
    Hm = hankel1(m, k * r)
    p_int += c_vec[i_m] * Jm * phase
    p_ext += (a_vec[i_m] * Jm + f_vec[i_m] * Hm) * phase

mask_in  = r <= R
mask_out = r >  R

p_total = np.zeros_like(X, dtype=complex)
p_total[mask_in]  = p_int[mask_in]
p_total[mask_out] = p_ext[mask_out]

# ==========================================
# 8. Рассеянное поле = p_total - p_bg ВЕЗДЕ
# ==========================================
p_sca = p_total - p_bg
p_plot = np.real(p_sca)

# ==========================================
# 9. График
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6))

vmax = np.percentile(np.abs(p_plot), 98)

im = ax.pcolormesh(X, Y, p_plot,
                   cmap='magma',
                   vmin=-vmax, vmax=vmax,
                   shading='auto')

# Резонатор
phi_full = np.linspace(0, 2 * np.pi, 720)
mask_arc = np.abs(((phi_full + np.pi) % (2 * np.pi)) - np.pi) > theta
phi_arc  = phi_full[mask_arc]
ax.plot(R * np.cos(phi_arc), R * np.sin(phi_arc), 'k-', linewidth=2.5)

ax.set_aspect('equal')
ax.set_xlim(-0.15, 0.15)
ax.set_ylim(-0.15, 0.15)
ax.set_xlabel('$x$ (m)', fontsize=12)
ax.set_ylabel('$y$ (m)', fontsize=12)
ax.set_title(f'Scattered field $p_t - p_b$,  $n={nb_inc}$,  $kR={kR}$', fontsize=13)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\mathrm{Re}(p_t - p_b)$', fontsize=12)

plt.tight_layout()
plt.show()