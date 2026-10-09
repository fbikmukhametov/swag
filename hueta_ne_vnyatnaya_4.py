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

# --- 4 резонатора в сетке 2x2 ---
centers = np.array([
    [-d/2, -d/2],   # 1: левый нижний
    [ d/2, -d/2],   # 2: правый нижний
    [-d/2,  d/2],   # 3: левый верхний
    [ d/2,  d/2],   # 4: правый верхний
])
theta0_list = [0.0, np.pi, 0.0, np.pi]   # вырез 1 и 3 -> вправо, 2 и 4 -> влево
N_res = len(centers)

nb_inc = 5

# ==========================================
# 2. Вспомогательные матрицы
# ==========================================
Jp    = np.diag(jvp(m_vals, kR))
Hp    = np.diag(jvp(m_vals, kR) + 1j * yvp(m_vals, kR))
J_m_R = np.diag(jv(m_vals, kR))
H_m_R = np.diag(jv(m_vals, kR) + 1j * yv(m_vals, kR))
I_n   = np.eye(n, dtype=complex)

# ==========================================
# 3. T-матрицы (своя для каждого theta0)
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

T_list  = []
PA_list = []
for th in theta0_list:
    T, PA = compute_T(th)
    T_list.append(T)
    PA_list.append(PA)

# ==========================================
# 4. Матрицы перехода p_ij (i,j = 0..3, i!=j)
# ==========================================
def compute_p(d_vec):
    dist  = np.linalg.norm(d_vec)
    angle = np.arctan2(d_vec[1], d_vec[0])
    P = np.zeros((n, n), dtype=complex)
    for i, q in enumerate(m_vals):
        for j, l in enumerate(m_vals):
            P[i, j] = hankel1(l - q, k * dist) * np.exp(1j * (l - q) * angle)
    return P

# p_mat[i][j]: перевод поля из j в i, вектор = r_i - r_j
p_mat = [[None]*N_res for _ in range(N_res)]
for i in range(N_res):
    for j in range(N_res):
        if i == j:
            continue
        p_mat[i][j] = compute_p(centers[i] - centers[j])

# ==========================================
# 5. Блочная матрица X (4n x 4n) и правая часть
# ==========================================
X = np.zeros((N_res*n, N_res*n), dtype=complex)
for i in range(N_res):
    for j in range(N_res):
        if i == j:
            block = I_n
        else:
            block = -T_list[i] @ p_mat[i][j]
        X[i*n:(i+1)*n, j*n:(j+1)*n] = block

# Падающее поле: глобальная мода J_nb(kr) e^{i nb φ}
# Разложение вокруг центра r_c:  a_m = J_{nb-m}(k|r_c|) * e^{i(nb-m) arg(r_c)}
a_list = []
for c in centers:
    rc   = np.linalg.norm(c)
    phic = np.arctan2(c[1], c[0])
    a_j  = np.array([jv(nb_inc - m, k * rc) * np.exp(1j * (nb_inc - m) * phic)
                     for m in m_vals])
    a_list.append(a_j)

rhs = np.concatenate([T_list[i] @ a_list[i] for i in range(N_res)])
b   = np.linalg.solve(X, rhs)
b_list = [b[i*n:(i+1)*n] for i in range(N_res)]

# ==========================================
# 6. Внутренние коэффициенты (с учётом поля от всех соседей)
# ==========================================
def compute_c(a_vec, b_vec, PA):
    M_int = PA @ J_m_R + (I_n - PA) @ Jp
    rhs_c = PA @ (J_m_R @ a_vec + H_m_R @ b_vec)
    return np.linalg.solve(M_int, rhs_c)

c_list = []
for i in range(N_res):
    a_local = a_list[i].copy()
    for j in range(N_res):
        if i != j:
            a_local = a_local + p_mat[i][j] @ b_list[j]
    c_list.append(compute_c(a_local, b_list[i], PA_list[i]))

# ==========================================
# 7. Сетка
# ==========================================
N_grid = 300
x = np.linspace(-0.02, 0.02, N_grid)
y = np.linspace(-0.02, 0.02, N_grid)
Xg, Yg = np.meshgrid(x, y)

# Падающее поле (глобальная цилиндрическая мода)
r_global   = np.sqrt(Xg**2 + Yg**2)
phi_global = np.arctan2(Yg, Xg)
p_inc = jv(nb_inc, k * r_global) * np.exp(1j * nb_inc * phi_global)

# --- поля каждого резонатора ---
p_int_fields = []
p_sca_fields = []
r_loc_list   = []
for i in range(N_res):
    dx = Xg - centers[i][0]
    dy = Yg - centers[i][1]
    r_loc   = np.sqrt(dx**2 + dy**2)
    phi_loc = np.arctan2(dy, dx)

    p_int = np.zeros_like(Xg, dtype=complex)
    p_sca = np.zeros_like(Xg, dtype=complex)
    for i_m, m in enumerate(m_vals):
        p_int += c_list[i][i_m] * jv(m, k*r_loc) * np.exp(1j*m*phi_loc)
        p_sca += b_list[i][i_m] * hankel1(m, k*r_loc) * np.exp(1j*m*phi_loc)

    p_int_fields.append(p_int)
    p_sca_fields.append(p_sca)
    r_loc_list.append(r_loc)

# --- маски ---
masks_in = [r_loc <= R for r_loc in r_loc_list]
mask_any_in = np.zeros_like(Xg, dtype=bool)
for mask in masks_in:
    mask_any_in |= mask
mask_out = ~mask_any_in

# --- сборка поля без двойного счёта ---
p_total = p_inc.copy()
for p_sca in p_sca_fields:
    p_total[mask_out] += p_sca[mask_out]
for i in range(N_res):
    p_total[masks_in[i]] = p_int_fields[i][masks_in[i]]

p_sca_total = p_total - p_inc
p_plot = np.real(p_sca_total)

# ==========================================
# 8. График
# ==========================================
fig, ax = plt.subplots(figsize=(8, 7))
vmax = np.abs(p_plot).max()
im = ax.pcolormesh(Xg, Yg, p_plot, cmap='bwr',
                   vmin=-vmax, vmax=vmax, shading='auto')

for c, th0 in zip(centers, theta0_list):
    phi_arc = np.linspace(th0 + theta, th0 + 2*np.pi - theta, 500)
    ax.plot(c[0] + R*np.cos(phi_arc), c[1] + R*np.sin(phi_arc),
            'k-', linewidth=2.5)

ax.set_aspect('equal')
ax.set_xlim(-0.02, 0.02)
ax.set_ylim(-0.02, 0.02)
ax.set_xlabel('$x$ (m)', fontsize=12)
ax.set_ylabel('$y$ (m)', fontsize=12)
ax.set_title(f'Scattered field, $n_b={nb_inc}$, $kR={kR}$, $N_{{res}}={N_res}$',
             fontsize=13)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\mathrm{Re}(p_t - p_{inc})$', fontsize=12)
plt.tight_layout()
plt.show()

print("Max |p_sca| (Python) =", np.abs(p_plot).max())
print("Min |p_sca| (Python) =", np.abs(p_plot).min())