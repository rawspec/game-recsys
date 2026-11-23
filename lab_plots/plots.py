# -*- coding: utf-8 -*-
"""Графики для ЛР (Приложение А): ЛАЧХ объекта, желаемая ЛАЧХ, ЛАЧХ регулятора, переходная характеристика."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["font.family"] = "DejaVu Sans"

# --- Исходные данные (вариант из приложения) ---
k_ob, k_iu = 50.0, 0.6
k = k_ob * k_iu                       # 30
T1, T2, T3 = 0.4, 0.2, 0.1
w1, w2, w3 = 1 / T1, 1 / T2, 1 / T3   # 2.5, 5, 10
w_sr = 2.2
w4, w5 = 0.3, 21.0
wa = 0.022
Ta = 1 / wa                           # 45.455
T4, T5 = 1 / w4, 1 / w5               # 3.333, 0.0476

H = lambda x: np.heaviside(x, 1.0)    # функция Хевисайда (Ф)

# --- А.1 Асимптотическая ЛАЧХ объекта ---
w = np.logspace(-2, np.log10(50), 2000)
L_ob = (20 * np.log10(k) - 20 * np.log10(w)
        - 20 * np.log10(w / w1) * H(w - w1)
        - 20 * np.log10(w / w2) * H(w - w2)
        - 20 * np.log10(w / w3) * H(w - w3))

# --- А.2 Желаемая ЛАЧХ ---
L_LF   = lambda w: 20 * np.log10(k) - 20 * np.log10(w)
L_conn = lambda w: L_LF(wa) - 40 * np.log10(w / wa)
L_mid  = lambda w: 0 - 20 * np.log10(w / w_sr)
L_HF1  = lambda w: L_mid(w1) - 40 * np.log10(w / w1)
L_HF2  = lambda w: L_HF1(w5) - 80 * np.log10(w / w5)

L_zhel = (L_LF(w) * H(wa - w)
          + L_conn(w) * H(w - wa) * H(w4 - w)
          + L_mid(w) * H(w - w4) * H(w1 - w)
          + L_HF1(w) * H(w - w1) * H(w5 - w)
          + L_HF2(w) * H(w - w5))

# --- А.3 ЛАЧХ регулятора ---
L_reg = L_zhel - L_ob

def setup_lachh(ax, title):
    ax.set_xlim(0.01, 50)
    ax.set_ylim(-40, 70)
    ax.set_xlabel("ω, рад/с")
    ax.set_ylabel("L, дБ")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)


# А.1. ЛАЧХ объекта
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogx(w, L_ob, "--", color="tab:blue", lw=2, label="L_ob(ω)")
setup_lachh(ax, "Асимптотическая ЛАЧХ объекта")
ax.legend()
fig.tight_layout()
fig.savefig("lab_plots/01_lachh_object.png", dpi=180)
plt.close(fig)

# А.2. Желаемая ЛАЧХ и ЛАЧХ объекта
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogx(w, L_ob, "--", color="tab:blue", lw=1.8, label="L_ob(ω) — объект")
ax.semilogx(w, L_zhel, "-", color="tab:red", lw=2, label="L_zhel(ω) — желаемая")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(w_sr, color="gray", ls=":", lw=1.2, label="ω_sr = 2,2 рад/с")
setup_lachh(ax, "Желаемая ЛАЧХ и ЛАЧХ объекта")
ax.legend()
fig.tight_layout()
fig.savefig("lab_plots/02_lachh_desired.png", dpi=180)
plt.close(fig)

# А.3. ЛАЧХ регулятора
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogx(w, L_reg, "-", color="tab:green", lw=2, label="L_reg(ω) = L_zhel(ω) − L_ob(ω)")
ax.axhline(0, color="black", ls="--", lw=1)
setup_lachh(ax, "ЛАЧХ регулятора")
ax.set_ylim(-30, 10)
ax.legend()
fig.tight_layout()
fig.savefig("lab_plots/03_lachh_controller.png", dpi=180)
plt.close(fig)

# --- А.4 Переходная характеристика замкнутой системы ---
# Через коэффициенты (раздел А.5)
num = [100.0, 30.0]
den = [0.01031, 0.5362, 8.886, 45.65, 1.0, 0.0]
den_cl = den.copy()
den_cl[4] += num[0]   # den + [0 0 0 0 num]
den_cl[5] += num[1]

sys = signal.TransferFunction(num, den_cl)
t = np.arange(0, 15.01, 0.01)
_, h = signal.step(sys, T=t)

h_ust = 1.0
delta = 0.05 * h_ust
h_max = h.max()
i_max = h.argmax()
sigma = (h_max - h_ust) / h_ust * 100
t_reg = t[np.flatnonzero(np.abs(h - h_ust) > delta)[-1]]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(t, h, "b-", lw=1.8, label="h(t)")
ax.axhline(h_ust, color="k", ls="--", lw=1, label="h_ust = 1")
ax.axhline(h_ust + delta, color="r", ls=":", lw=1.2, label="Коридор ±5 %")
ax.axhline(h_ust - delta, color="r", ls=":", lw=1.2)
ax.axvline(t[i_max], color="gray", ls=":", lw=1)
ax.text(t[i_max] + 0.2, h_max, f"h_max = {h_max:.3f}\nσ = {sigma:.1f} %", fontsize=9)
ax.set_xlim(0, 15)
ax.set_ylim(0, max(1.3, h_max * 1.1))
ax.set_xlabel("t, с")
ax.set_ylabel("h(t)")
ax.set_title("Переходная характеристика замкнутой системы с регулятором")
ax.grid(True, ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig("lab_plots/04_step_response.png", dpi=180)
plt.close(fig)

print(f"h_ust = {h_ust}, h_max = {h_max:.4f}, sigma = {sigma:.2f} %, "
      f"t_max = {t[i_max]:.2f} c, t_reg = {t_reg:.2f} c")
