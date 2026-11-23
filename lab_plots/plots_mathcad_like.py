# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots-matchad-like"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "mathtext.fontset": "stix",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "black",
    "axes.labelsize": 11,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

k = 50.0 * 0.6
w1, w2, w3 = 2.5, 5.0, 10.0
w_sr = 2.2
w4, w5 = 0.3, 21.0
wa = 0.022


def heaviside(x):
    return np.heaviside(x, 1.0)


def l_lf(w):
    return 20 * np.log10(k) - 20 * np.log10(w)


def l_conn(w):
    return l_lf(wa) - 40 * np.log10(w / wa)


def l_mid(w):
    return -20 * np.log10(w / w_sr)


def l_hf1(w):
    return l_mid(w1) - 40 * np.log10(w / w1)


def l_hf2(w):
    return l_hf1(w5) - 80 * np.log10(w / w5)


w = np.logspace(-2, np.log10(50), 2000)
l_ob = (l_lf(w)
        - 20 * np.log10(w / w1) * heaviside(w - w1)
        - 20 * np.log10(w / w2) * heaviside(w - w2)
        - 20 * np.log10(w / w3) * heaviside(w - w3))
l_zhel = (l_lf(w) * heaviside(wa - w)
          + l_conn(w) * heaviside(w - wa) * heaviside(w4 - w)
          + l_mid(w) * heaviside(w - w4) * heaviside(w1 - w)
          + l_hf1(w) * heaviside(w - w1) * heaviside(w5 - w)
          + l_hf2(w) * heaviside(w - w5))
l_reg = l_zhel - l_ob


def setup_lachh(ax, ylim=(-40, 70)):
    from matplotlib.ticker import FormatStrFormatter, NullLocator

    ax.set_xlim(0.01, 50)
    ax.set_ylim(*ylim)
    ax.set_xlabel("ω, рад/с", labelpad=6)
    ax.set_ylabel("L, дБ", labelpad=6)
    ax.grid(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(which="major", length=4, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, filename):
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.95)
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 4.6))
ax.semilogx(w, l_ob, color="#0000ff", linestyle="--", linewidth=1.4)
setup_lachh(ax)
save(fig, "01_lachh_object.png")

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.semilogx(w, l_ob, color="#0000ff", linestyle="--", linewidth=1.3)
ax.semilogx(w, l_zhel, color="#ff0000", linestyle="-", linewidth=1.4)
ax.axhline(0, color="black", linewidth=0.65)
ax.axvline(w_sr, color="#666666", linestyle=":", linewidth=0.9)
setup_lachh(ax)
save(fig, "02_lachh_desired.png")

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.semilogx(w, l_reg, color="#008000", linestyle="-", linewidth=1.5)
ax.axhline(0, color="black", linestyle="--", linewidth=0.75)
setup_lachh(ax, ylim=(-30, 10))
save(fig, "03_lachh_controller.png")

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.semilogx(w, l_ob, color="#0000ff", linestyle="--", linewidth=1.3)
ax.semilogx(w, l_zhel, color="#ff0000", linestyle="-", linewidth=1.4)
ax.semilogx(w, l_reg, color="#008000", linestyle="-", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.65)
ax.axvline(w_sr, color="#666666", linestyle=":", linewidth=0.9)
setup_lachh(ax)
save(fig, "lachh.png")

num = [100.0, 30.0]
den = [0.01031, 0.5362, 8.886, 45.65, 1.0, 0.0]
den_cl = den.copy()
den_cl[4] += num[0]
den_cl[5] += num[1]
t = np.arange(0, 15.01, 0.01)
_, h = signal.step(signal.TransferFunction(num, den_cl), T=t)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(t, h, color="#0000ff", linewidth=1.4)
ax.axhline(1, color="black", linestyle="--", linewidth=0.75)
ax.axhline(1.05, color="#ff0000", linestyle=":", linewidth=0.9)
ax.axhline(0.95, color="#ff0000", linestyle=":", linewidth=0.9)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.3)
ax.set_xlabel("t, с", labelpad=6)
ax.set_ylabel("h(t)", labelpad=6)
ax.grid(False)
ax.tick_params(which="major", length=4, width=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.95)
fig.savefig(OUTPUT_DIR / "04_step_response.png", dpi=180)
fig.savefig(OUTPUT_DIR / "step.png", dpi=180)
plt.close(fig)

print(f"Графики сохранены в {OUTPUT_DIR}")
