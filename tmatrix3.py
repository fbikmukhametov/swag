import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jvp, yvp, hankel2

# ==========================================
# 1. Параметры
# ==========================================
M         = 30
m_vals    = np.arange(-M, M + 1)
theta_deg = 10                     # ЗАКРЫТЫЙ цилиндр
theta     = np.deg2rad(theta_deg)
theta0    = 0

R         = 0.004
r0        = 0.012
N_phi     = 359
kR_target = 5.1291

file_path = r"C:\Users\bikfa\Desktop\comsol_data\data_nb{nb}_kR{kR}.txt"

# ==========================================
# 2. ОБЩАЯ аналитическая T (формула 6)
#    T = T_HW + W_out · K^{-1} · W_in
#    При theta=0 второе слагаемое само обнуляется,
#    но формула та же — используем pinv(K) для устойчивости.
# ==========================================
def compute_T_analytical(kR, lam_rel=1e-4):
    Jp = np.diag(jvp(m_vals, kR))
    Hp = np.diag(jvp(m_vals, kR) - 1j*yvp(m_vals, kR))

    n = len(m_vals)
    PA = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            m, mp = m_vals[i], m_vals[j]
            if i == j:
                PA[i, j] = theta / np.pi
            else:
                PA[i, j] = (np.sin((m - mp)*theta) / (np.pi*(m - mp))) \
                           * np.exp(1j*(m - mp)*theta0)

    Hp_inv   = np.diag(1.0 / np.diag(Hp))
    HpJp_inv = np.diag(1.0 / np.diag(Hp @ Jp))
    T_HW  = -Hp_inv @ Jp
    K     = PA @ HpJp_inv @ PA
    W_out = Hp_inv @ PA
    W_in  = PA @ Hp_inv


    U, s, Vh = np.linalg.svd(K)
    print(f"  s/s[0] = {s / s[0]}")
    print(f"  rank(K), порог 1e-6: {(s > 1e-6*s[0]).sum()}")
    
    if theta == 0 or np.linalg.norm(K) < 1e-15:
        return T_HW

    lam = 3e-4 * np.linalg.norm(K)
    X = np.linalg.solve(K + lam * np.eye(n, dtype=complex), W_in)
    #X=np.linalg.solve(K, W_in)
    return T_HW + W_out @ X
# ==========================================
# 3. COMSOL T на kR = 4.0
# ==========================================
data = np.loadtxt(file_path, delimiter='|', comments='%')
nb_all = data[:, 2].astype(int)
kR_all = data[:, 3]
p_t_re = data[:, 7]; p_t_im = data[:, 8]
p_b_re = data[:, 9]; p_b_im = data[:, 10]

nb_vals = np.unique(nb_all)
kR_vals = np.unique(kR_all)
i_kR    = np.argmin(np.abs(kR_vals - kR_target))
kR_val  = kR_vals[i_kR]
k       = kR_val / R

print(f"kR = {kR_val:.4f}, nb = {nb_vals}")

T_comsol = np.zeros((len(m_vals), len(nb_vals)), dtype=complex)
for i_nb, nb in enumerate(nb_vals):
    mask = (nb_all == nb) & np.isclose(kR_all, kR_val)
    if mask.sum() != N_phi:
        continue
    p_t   = p_t_re[mask] + 1j*p_t_im[mask]
    p_b   = p_b_re[mask] + 1j*p_b_im[mask]
    p_sca = p_t - p_b

    P_all  = np.fft.fftshift(np.fft.fft(p_sca)) / N_phi
    center = N_phi // 2
    P_m    = P_all[center - M : center + M + 1]

    H_m = hankel2(m_vals, k * r0)
    T_comsol[:, i_nb] = P_m / H_m

# ==========================================
# 4. Аналитика через общую формулу
# ==========================================
T_anal = compute_T_analytical(kR_val)

# ==========================================
# 5. Таблица
# ==========================================
print(f"\n=== Диагональные элементы (kR = {kR_val:.3f}) ===")
print(f"{'m':>3} | {'|Tc|':>12} | {'|Ta|':>12} | {'|Tc - Ta|':>12}")
print("-" * 50)
for i_m, m in enumerate(m_vals):
    if m in nb_vals:
        i_nb = list(nb_vals).index(m)
        tc = T_comsol[i_m, i_nb]
        ta = T_anal[i_m, i_nb]
        print(f"{m:>3} | {abs(tc):>12.6e} | {abs(ta):>12.6e} | "
              f"{abs(tc-ta):>12.6e}")

# ==========================================
# 6. Три тепловые карты (ЛИНЕЙНАЯ шкала!)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- (1) COMSOL ---
T1 = np.abs(T_comsol)
pcm1 = axes[0].pcolormesh(nb_vals, m_vals, T1,
                          cmap='RdBu_r', shading='nearest')
axes[0].set_xlabel('incident $n$')
axes[0].set_ylabel('outgoing $m$')
axes[0].set_title(f'|T| COMSOL,  $kR = {kR_val:.3f}$')
axes[0].set_aspect('equal')
plt.colorbar(pcm1, ax=axes[0], label=r'$|T_{mn}|$')

# --- (2) Аналитика ---
T2 = np.abs(T_anal)
pcm2 = axes[1].pcolormesh(m_vals, m_vals, T2,
                          cmap='RdBu_r', shading='nearest')
axes[1].set_xlabel('incident $n$')
axes[1].set_ylabel('outgoing $m$')
axes[1].set_title(f'|T| аналитика,  $kR = {kR_val:.3f}$')
axes[1].set_aspect('equal')
plt.colorbar(pcm2, ax=axes[1], label=r'$|T_{mn}|$')

# --- (3) Модуль комплексной разности ---
T3 = np.abs(T_comsol - T_anal)
pcm3 = axes[2].pcolormesh(nb_vals, m_vals, T3,
                          cmap='RdBu_r', shading='nearest')
axes[2].set_xlabel('incident $n$')
axes[2].set_ylabel('outgoing $m$')
axes[2].set_title(r'$|T^{\mathrm{COMSOL}}_{mn} - T^{\mathrm{anal}}_{mn}|$')
axes[2].set_aspect('equal')
plt.colorbar(pcm3, ax=axes[2], label=r'$|\Delta T_{mn}|$')

plt.tight_layout()
plt.show()

# ==========================================
# 7. Сохранение
# ==========================================
np.save(r"C:\Users\bikfa\Desktop\comsol_data\T_comsol_kR4.npy", T_comsol)
np.save(r"C:\Users\bikfa\Desktop\comsol_data\T_anal_kR4.npy",   T_anal)
print("Сохранено: T_comsol_kR4.npy, T_anal_kR4.npy")

import numpy as np
# ... (предполагаем, что T_comsol, T_anal, m_vals, nb_vals уже посчитаны)

# Копируем аналитику в квадратную форму M×M (столбцы = nb, строки = m)
T_anal_sq = np.zeros((len(m_vals), len(m_vals)), dtype=complex)
for i_nb, nb in enumerate(nb_vals):
    T_anal_sq[:, i_nb] = T_anal[:, i_nb]

# Маска: диагональ и вне диагонали
mask_diag = np.eye(len(m_vals), dtype=bool)
mask_off  = ~mask_diag

# Относительная ошибка на диагонали
err_diag = np.abs(T_comsol[mask_diag] - T_anal_sq[mask_diag]) / \
           np.abs(T_anal_sq[mask_diag])
# Абсолютная ошибка вне диагонали
err_off  = np.abs(T_comsol[mask_off] - T_anal_sq[mask_off])

print("=== Диагональ ===")
print(f"  средняя |T|       : {np.mean(np.abs(T_anal_sq[mask_diag])):.3e}")
print(f"  средняя отн. ошиб.: {np.mean(err_diag):.3e}")
print(f"  макс. отн. ошиб.  : {np.max(err_diag):.3e}")

print("\n=== Вне диагонали ===")
print(f"  средняя |T|       : {np.mean(np.abs(T_anal_sq[mask_off])):.3e}")
print(f"  средняя |ΔT|      : {np.mean(err_off):.3e}")
print(f"  макс. |ΔT|        : {np.max(err_off):.3e}")
print(f"  отношение ΔT/|T|  : {np.mean(err_off) / np.mean(np.abs(T_anal_sq[mask_off])):.3f}")

print(f"\n=== Подбор lam_rel ===")
for lam_rel in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]:
    T_test = compute_T_analytical(kR_val, lam_rel=lam_rel)
    # Диагональ
    print(f"\nlam_rel = {lam_rel:.0e}")
    for i_m, m in enumerate(m_vals):
        if m in nb_vals:
            i_nb = list(nb_vals).index(m)
            tc = abs(T_comsol[i_m, i_nb])
            ta = abs(T_test[i_m, i_nb])
            print(f"  m={m:>3}: |Ta|={ta:.4e}  |Tc|={tc:.4e}  err={abs(tc-ta)/tc*100:.2f}%")

# ==========================================
# 8. ОБРЕЗКА ДО ВНУТРЕННИХ МОД
# ==========================================
M_crop = 30                                    # сколько мод оставить с каждой стороны

mask_m  = np.abs(m_vals)  <= M_crop           # маска строк (моды m)
mask_nb = np.abs(nb_vals) <= M_crop           # маска столбцов (моды nb)

# --- COMSOL (уже в форме M×len(nb_vals)) ---
T_comsol_crop = T_comsol[np.ix_(mask_m, mask_nb)]

# --- Аналитика: сначала выровнять по nb_vals, потом обрезать ---
T_anal_sq = np.zeros((len(m_vals), len(nb_vals)), dtype=complex)
for i_nb, nb in enumerate(nb_vals):
    i_m_global = list(m_vals).index(nb)       # индекс nb в полном m_vals
    T_anal_sq[:, i_nb] = T_anal[:, i_m_global]

T_anal_crop = T_anal_sq[np.ix_(mask_m, mask_nb)]

# --- Разница ---
T_diff_crop = np.abs(T_comsol_crop - T_anal_crop)

# --- Оси для графиков ---
m_crop  = m_vals[mask_m]                      # [-3 -2 -1 0 1 2 3]
nb_crop = nb_vals[mask_nb]                    # [-3 -2 -1 0 1 2 3]

print(f"\n=== После обрезки: {len(m_crop)} × {len(nb_crop)} ===")

# ==========================================
# 9. ТРИ ГРАФИКА С ОБРЕЗКОЙ
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

pcm1 = axes[0].pcolormesh(nb_crop, m_crop, np.abs(T_comsol_crop),
                          cmap='RdBu_r', shading='nearest')
axes[0].set_title(f'|T| COMSOL,  $kR = {kR_val:.3f}$')
axes[0].set_xlabel('incident $n$'); axes[0].set_ylabel('outgoing $m$')
axes[0].set_aspect('equal')
plt.colorbar(pcm1, ax=axes[0], label=r'$|T_{mn}|$')

pcm2 = axes[1].pcolormesh(nb_crop, m_crop, np.abs(T_anal_crop),
                          cmap='RdBu_r', shading='nearest')
axes[1].set_title(f'|T| аналитика,  $kR = {kR_val:.3f}$')
axes[1].set_xlabel('incident $n$'); axes[1].set_ylabel('outgoing $m$')
axes[1].set_aspect('equal')
plt.colorbar(pcm2, ax=axes[1], label=r'$|T_{mn}|$')

pcm3 = axes[2].pcolormesh(nb_crop, m_crop, T_diff_crop,
                          cmap='RdBu_r', shading='nearest')
axes[2].set_title(r'$|T^{\mathrm{COMSOL}}_{mn} - T^{\mathrm{anal}}_{mn}|$')
axes[2].set_xlabel('incident $n$'); axes[2].set_ylabel('outgoing $m$')
axes[2].set_aspect('equal')
plt.colorbar(pcm3, ax=axes[2], label=r'$|\Delta T_{mn}|$')

plt.tight_layout()
plt.show()

# ==========================================
# 10. СТАТИСТИКА ПО ОБРЕЗАННОМУ БЛОКУ
# ==========================================
mask_diag = np.eye(len(m_crop), dtype=bool)
mask_off  = ~mask_diag

err_diag = np.abs(T_comsol_crop[mask_diag] - T_anal_crop[mask_diag]) / \
           np.abs(T_anal_crop[mask_diag])
err_off  = np.abs(T_comsol_crop[mask_off] - T_anal_crop[mask_off])

print(f"\n=== После обрезки до |m| ≤ {M_crop} ===")
print(f"Диагональ:")
print(f"  средняя отн. ошиб.: {np.mean(err_diag):.3e}")
print(f"  макс. отн. ошиб.  : {np.max(err_diag):.3e}")
print(f"Вне диагонали:")
print(f"  средняя |ΔT|      : {np.mean(err_off):.3e}")
print(f"  макс. |ΔT|        : {np.max(err_off):.3e}")

# Диагональная таблица по обрезанным модам
print(f"\n=== Диагональ после обрезки ===")
print(f"{'m':>3} | {'|Tc|':>12} | {'|Ta|':>12} | {'err, %':>8}")
print("-" * 45)
for i_m, m in enumerate(m_crop):
    tc = abs(T_comsol_crop[i_m, i_m])
    ta = abs(T_anal_crop[i_m, i_m])
    err = abs(tc - ta) / ta * 100
    print(f"{m:>3} | {tc:>12.6e} | {ta:>12.6e} | {err:>8.2f}")
import numpy as np
data = np.loadtxt(file_path, delimiter='|', comments='%')
print(f"shape = {data.shape}")
print(f"kR_vals = {np.unique(data[:, 3])}")
print(f"freq_vals = {np.unique(data[:, 4])}")
print(f"max|p_t| = {np.max(np.abs(data[:,7] + 1j*data[:,8])):.3e}")
print(f"max|p_b| = {np.max(np.abs(data[:,9] + 1j*data[:,10])):.3e}")

print("=" * 60)
print("Диагональ (m = n) и антидиагональ (m = -n)")
print("=" * 60)

n_modes = len(m_vals)
for i in range(n_modes):
    m     = m_vals[i]
    j_diag = i
    j_anti = n_modes - 1 - i

    tc_diag = T_comsol[i, j_diag] if j_diag < T_comsol.shape[1] else 0
    ta_diag = T_anal_sq[i, j_diag] if j_diag < T_anal_sq.shape[1] else 0
    tc_anti = T_comsol[i, j_anti] if j_anti < T_comsol.shape[1] else 0
    ta_anti = T_anal_sq[i, j_anti] if j_anti < T_anal_sq.shape[1] else 0

    print(f"m={m:>3} | "
          f"diag: Tc={abs(tc_diag):.3e} Ta={abs(ta_diag):.3e} | "
          f"anti: Tc={abs(tc_anti):.3e} Ta={abs(ta_anti):.3e}")

# Проверка: заполнены ли обе матрицы корректно
print(f"m_vals = {m_vals}")
print(f"nb_vals = {nb_vals}")
print(f"len(m_vals) = {len(m_vals)}")
print(f"len(nb_vals) = {len(nb_vals)}")
