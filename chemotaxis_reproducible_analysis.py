#!/usr/bin/env python3
"""
Reproducible quantitative analysis for:
"Quantifying transient biochemical information in Escherichia coli chemotaxis"

Purpose
-------
This script reproduces the principal quantitative results of the reconstructed
manuscript and revised Supplementary Information from one consistent pipeline.

What is recomputed here from source/model inputs
------------------------------------------------
1) Formal microscopic capacity and total-count coarse-graining.
2) Shimizu et al. endpoint-based fixed-modification calibration:
   alpha, m0, and G_m^A.
3) Within-state bootstrap precision and separate structural sensitivity.
4) Ligand-to-methylation mapping m*(L) and logarithmic susceptibility kappa(L).
5) Colin-derived effective methylation-coordinate noise.
6) Static mutual information I(L;m), N_eff, and environmental-prior sensitivity.
7) Run-scale ligand changes and the environmental traversal timescale tau_env.
8) Mattingly kinase benchmark beta g^2 and rate-scale ratios.
9) Table S3 boundary-treatment ratios and corrected Figure S5.

What is NOT independently reconstructed from raw trajectories here
------------------------------------------------------------------
The final open-domain dynamic methylation-information-rate medians and Monte
Carlo intervals are included as the manuscript's validated final outputs:
    g=0.1: 1.74e-4 [1.52e-4, 2.07e-4]
    g=0.2: 6.67e-4 [5.85e-4, 7.97e-4]
    g=0.3: 1.42e-3 [1.24e-3, 1.69e-3]
    g=0.4: 2.40e-3 [2.11e-3, 2.87e-3] bits/s.
A full independent regeneration of those spectral Monte Carlo values requires
the original trajectory/spectrum simulation implementation or raw trajectory
data. This script deliberately does not invent that missing layer.

Boundary-condition note
-----------------------
Historical Table S3 absolute rates came from an earlier calibration. The revised
SI reports only boundary-treatment ratios relative to the open-domain model.
Those ratios isolate boundary sensitivity. Calibration-independence is exact
under a common multiplicative recalibration in the weak-SNR regime; if a future
full-spectrum check shows SNR ~ 1 or larger for a boundary treatment, the ratios
should be recomputed from that full model.

Run
---
    python chemotaxis_reproducible_analysis.py

Dependencies
------------
numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SHIMIZU_CSV = DATA / "44320_2010_BFMSB201037_MOESM10_ESM.csv"

# -------------------------
# Declared model parameters
# -------------------------
N_MCP = 15000
N_MWC = 6.0
A0 = 1.0 / 3.0

KI_mM = 0.018
KA_mM = 2.903
KI_uM = KI_mM * 1000.0
KA_uM = KA_mM * 1000.0

LAMBDA_FRET = 0.10

# Colin-derived effective relative-activity/FRET-noise anchors used in the manuscript.
# Buffer is plotted at 1 uM only as a display proxy on a logarithmic axis.
COLIN_BUFFER_VAR_RELATIVE_ACTIVITY = 0.46
COLIN_SIGMA_A = {
    "buffer": math.sqrt(COLIN_BUFFER_VAR_RELATIVE_ACTIVITY),
    "10uM": 0.405,
    "25uM": 0.363,
}
NOISE_L_ANCHORS_uM = np.array([1.0, 10.0, 25.0])

PRIOR_MAIN_uM = (1.0, 1.0e4)
PRIOR_WINDOWS_uM = [
    (0.3, 3.0e3),
    (1.0, 1.0e4),
    (3.0, 3.0e4),
    (10.0, 1.0e5),
    (18.0, 1.8e5),
]

GRADIENTS = np.array([0.1, 0.2, 0.3, 0.4])
V_MM_S = 0.020
TAU_RUN_S = 1.0
TAU_M_S = 10.0
BETA_KINASE = 0.22  # bits s^-1 mm^2

# Final open-domain model outputs retained exactly as reported in the manuscript/SI.
OPEN_RATE = np.array([1.74e-4, 6.67e-4, 1.42e-3, 2.40e-3])
OPEN_RATE_LO = np.array([1.52e-4, 5.85e-4, 1.24e-3, 2.11e-3])
OPEN_RATE_HI = np.array([2.07e-4, 7.97e-4, 1.69e-3, 2.87e-3])

# Historical pre-final-calibration boundary outputs used only to form dimensionless ratios.
BOUNDARY_OLD = {
    "reflecting": np.array([0.00679, 0.01216, 0.01744, 0.01978]),
    "finite_window": np.array([0.00382, 0.01027, 0.01548, 0.02057]),
    "open": np.array([0.00120, 0.00461, 0.00981, 0.01658]),
    "reinjection_stress_test": np.array([0.0092, 0.0245, 0.0391, 0.0648]),
}


# =====================================
# 1. Formal capacities / coarse-graining
# =====================================
def formal_capacities():
    H_micro = 4.0 * N_MCP
    H_count = math.log2(4 * N_MCP + 1)
    return H_micro, H_count


# =====================================
# 2. Shimizu endpoint calibration
# =====================================
def load_shimizu_source(path=SHIMIZU_CSV):
    raw = pd.read_csv(path, header=None)
    headers = list(raw.iloc[1])
    states = {}
    for j in range(0, raw.shape[1], 2):
        if pd.isna(headers[j]):
            continue
        state = str(headers[j]).split(":")[0]
        x = pd.to_numeric(raw.iloc[2:, j], errors="coerce").to_numpy(float)
        y = pd.to_numeric(raw.iloc[2:, j + 1], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        states[state] = (x[mask], y[mask])
    scale = max(np.nanmax(y) for _, y in states.values())
    return states, scale


def f_L(L_mM):
    L_mM = np.asarray(L_mM, dtype=float)
    return np.log((1.0 + L_mM / KI_mM) / (1.0 + L_mM / KA_mM))


def activity_mwc(L_mM, fm, N=N_MWC):
    return 1.0 / (1.0 + np.exp(N * (f_L(L_mM) + fm)))


def fit_fm(x_mM, y_raw, scale):
    y = np.asarray(y_raw, dtype=float) / scale
    res = minimize_scalar(
        lambda fm: np.mean((activity_mwc(x_mM, fm) - y) ** 2),
        bounds=(-10, 10),
        method="bounded",
    )
    return float(res.x), float(res.fun)


def fit_all_fm(states, scale):
    wanted = ["QEEE", "QEQE", "QEQQ", "QQQQ", "QEmQQ", "QEmQEm"]
    rows = []
    for state in wanted:
        fm, mse = fit_fm(*states[state], scale)
        rows.append({
            "state": state,
            "fm": fm,
            "mse_activity": mse,
            "n_points": len(states[state][0]),
        })
    return pd.DataFrame(rows)


def endpoint_calibration(fm_df):
    fm = dict(zip(fm_df["state"], fm_df["fm"]))

    # Family 1: E -> Q, transition counts 1..4
    q_states = ["QEEE", "QEQE", "QEQQ", "QQQQ"]
    xq = np.arange(1, 5, dtype=float)
    yq = np.array([fm[s] for s in q_states])
    qfit = linregress(xq, yq)

    # Family 2: Q -> Em, transition counts 0..2
    em_states = ["QQQQ", "QEmQQ", "QEmQEm"]
    xe = np.arange(0, 3, dtype=float)
    ye = np.array([fm[s] for s in em_states])
    efit = linregress(xe, ye)

    # Extrapolated physical endpoints: native EEEE (m=0), fully methylated Em4 (m=4)
    f_EEEE = qfit.intercept
    f_Em4 = efit.intercept + 4.0 * efit.slope
    alpha = (f_EEEE - f_Em4) / 4.0
    m0 = f_EEEE / alpha
    GmA = N_MWC * alpha * A0 * (1.0 - A0)

    return {
        "q_states": q_states,
        "xq": xq,
        "yq": yq,
        "qfit": qfit,
        "q_resid": yq - (qfit.intercept + qfit.slope * xq),
        "em_states": em_states,
        "xe": xe,
        "ye": ye,
        "efit": efit,
        "em_resid": ye - (efit.intercept + efit.slope * xe),
        "f_EEEE_extrap": f_EEEE,
        "f_Em4_extrap": f_Em4,
        "alpha": alpha,
        "m0": m0,
        "GmA": GmA,
    }


def bootstrap_calibration(states, scale, fm_df, cal, n=10000, seed=20260915):
    rng = np.random.default_rng(seed)
    base_fm = dict(zip(fm_df["state"], fm_df["fm"]))
    all_states = ["QEEE", "QEQE", "QEQQ", "QQQQ", "QEmQQ", "QEmQEm"]

    # Within-state residual bootstrap: experimental precision conditional on state identity.
    within = np.empty((n, 2))
    for b in range(n):
        fb = {}
        for state in all_states:
            x, yraw = states[state]
            y = yraw / scale
            pred = activity_mwc(x, base_fm[state])
            resid = y - pred
            yb = (pred + rng.choice(resid, size=len(resid), replace=True)) * scale
            fb[state] = fit_fm(x, yb, scale)[0]

        cq = np.polyfit(cal["xq"], [fb[s] for s in cal["q_states"]], 1)
        ce = np.polyfit(cal["xe"], [fb[s] for s in cal["em_states"]], 1)
        f0 = cq[1]
        f4 = ce[0] * 4.0 + ce[1]
        a = (f0 - f4) / 4.0
        within[b] = [a, f0 / a]

    # Structural bootstrap: state-to-state scatter around the two family trends.
    qpred = cal["qfit"].intercept + cal["qfit"].slope * cal["xq"]
    epred = cal["efit"].intercept + cal["efit"].slope * cal["xe"]
    structural = np.empty(n)
    for b in range(n):
        yqb = qpred + rng.choice(cal["q_resid"], size=len(cal["q_resid"]), replace=True)
        yeb = epred + rng.choice(cal["em_resid"], size=len(cal["em_resid"]), replace=True)
        cq = np.polyfit(cal["xq"], yqb, 1)
        ce = np.polyfit(cal["xe"], yeb, 1)
        f0 = cq[1]
        f4 = ce[0] * 4.0 + ce[1]
        structural[b] = (f0 - f4) / 4.0

    return within, structural


# =====================================
# 3. Static mapping / noise / MI
# =====================================
def make_static_functions(alpha, m0, GmA):
    def mstar(L_uM):
        L = np.asarray(L_uM, dtype=float)
        return m0 + (1.0 / alpha) * np.log(
            (1.0 + L / KI_uM) / (1.0 + L / KA_uM)
        )

    def kappa(L_uM):
        L = np.asarray(L_uM, dtype=float)
        return (1.0 / alpha) * (
            L / (KI_uM + L) - L / (KA_uM + L)
        )

    sigmaA_anchors = np.array([
        COLIN_SIGMA_A["buffer"],
        COLIN_SIGMA_A["10uM"],
        COLIN_SIGMA_A["25uM"],
    ])

    def sigma_A(L_uM):
        L = np.asarray(L_uM, dtype=float)
        return np.interp(
            np.log10(L),
            np.log10(NOISE_L_ANCHORS_uM),
            sigmaA_anchors,
            left=sigmaA_anchors[0],
            right=sigmaA_anchors[-1],
        )

    def sigma_R(L_uM):
        return LAMBDA_FRET * sigma_A(L_uM)

    def sigma_m(L_uM):
        # Algebraically equivalent to sigma_R/(lambda*GmA)
        return sigma_A(L_uM) / GmA

    return mstar, kappa, sigma_A, sigma_R, sigma_m


def mutual_information(alpha, m0, GmA, Lmin=1.0, Lmax=1e4, nL=700, nm=2200):
    mstar, _, _, _, sigma_m = make_static_functions(alpha, m0, GmA)

    # log-uniform prior -> uniform in z = ln L
    z = np.linspace(np.log(Lmin), np.log(Lmax), nL)
    L = np.exp(z)
    mu = mstar(L)
    sig = sigma_m(L)

    mg = np.linspace(np.min(mu - 6 * sig), np.max(mu + 6 * sig), nm)
    dm = mg[1] - mg[0]
    P = np.exp(-0.5 * ((mg[:, None] - mu[None, :]) / sig[None, :]) ** 2)
    P /= np.sqrt(2 * np.pi) * sig[None, :]
    pm = P.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(P > 0, P * np.log2(P / pm[:, None]), 0.0)

    I = float((kl.sum(axis=0) * dm).mean())
    return I, 2.0 ** I


# =====================================
# 4. Run-scale and timescale calculations
# =====================================
def run_scale_changes():
    dlnL = GRADIENTS * V_MM_S * TAU_RUN_S
    return pd.DataFrame({
        "g_mm^-1": GRADIENTS,
        "delta_lnL_one_run": dlnL,
        "delta_lnL_binary_units": dlnL / math.log(2),
    })


def timescale_table(I_main):
    Neff = 2.0 ** I_main
    full_log_span = math.log(PRIOR_MAIN_uM[1] / PRIOR_MAIN_uM[0])
    one_interval = full_log_span / Neff
    tau_env_s = one_interval / (GRADIENTS * V_MM_S)
    eps = TAU_M_S / tau_env_s
    return pd.DataFrame({
        "g_mm^-1": GRADIENTS,
        "tau_env_s": tau_env_s,
        "tau_env_min": tau_env_s / 60.0,
        "epsilon_ad": eps,
    })


# =====================================
# 5. Dynamic benchmark and boundary ratios
# =====================================
def dynamic_summary():
    kinase = BETA_KINASE * GRADIENTS ** 2
    ratio_pct = 100.0 * OPEN_RATE / kinase
    return pd.DataFrame({
        "g_mm^-1": GRADIENTS,
        "methylation_open_median_bits_s": OPEN_RATE,
        "methylation_open_lo_bits_s": OPEN_RATE_LO,
        "methylation_open_hi_bits_s": OPEN_RATE_HI,
        "kinase_measured_bits_s": kinase,
        "rate_scale_ratio_percent": ratio_pct,
    })


def boundary_ratios():
    op = BOUNDARY_OLD["open"]
    return pd.DataFrame({
        "g_mm^-1": GRADIENTS,
        "reflecting_over_open": BOUNDARY_OLD["reflecting"] / op,
        "finite_window_over_open": BOUNDARY_OLD["finite_window"] / op,
        "reinjection_over_open_stress_test": BOUNDARY_OLD["reinjection_stress_test"] / op,
    })


# =====================================
# 6. Figures
# =====================================
def make_figures(alpha, m0, GmA, cal, prior_df, bnd_df, times_df, dyn_df):
    mstar, kappa, _, _, sigma_m = make_static_functions(alpha, m0, GmA)

    # Endpoint calibration
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(cal["xq"], cal["yq"], label="E→Q family")
    ax.scatter(cal["xe"], cal["ye"], marker="s", label="Q→Em family")
    xx = np.linspace(0, 4, 200)
    ax.plot(xx, cal["qfit"].intercept + cal["qfit"].slope * xx, label="E→Q fit")
    ax.plot(xx, cal["efit"].intercept + cal["efit"].slope * xx, label="Q→Em fit")
    ax.scatter([0, 4], [cal["f_EEEE_extrap"], cal["f_Em4_extrap"]],
               marker="x", s=80, label="Extrapolated physical endpoints")
    ax.set_xlabel("Transition count within each modification family")
    ax.set_ylabel(r"Fitted modification free energy $f_m$")
    ax.set_title(f"Endpoint calibration: α={alpha:.3f}, m₀={m0:.3f}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "shimizu_endpoint_calibration.png", dpi=250)
    plt.close(fig)

    # m*(L), kappa
    L = np.logspace(-1, 4, 500)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(L, mstar(L))
    ax.set_xscale("log")
    ax.set_xlabel("MeAsp concentration, L (μM)")
    ax.set_ylabel("Adapted methylation coordinate, m*(L)")
    ax.set_title("Ligand-to-methylation mapping")
    fig.tight_layout()
    fig.savefig(OUT / "ligand_to_methylation.png", dpi=250)
    plt.close(fig)

    # Figure 3A-style conditional distributions
    Lrep = np.array([10, 30, 100, 300, 1000.0])
    mu = mstar(Lrep)
    sig = sigma_m(Lrep)
    mg = np.linspace(np.min(mu - 5 * sig), np.max(mu + 5 * sig), 1400)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for Li, mi, si in zip(Lrep, mu, sig):
        p = np.exp(-0.5 * ((mg - mi) / si) ** 2) / (np.sqrt(2*np.pi) * si)
        ax.plot(mg, p, label=f"{Li:g} μM")
    ax.set_xlabel("Collective methylation coordinate, m")
    ax.set_ylabel("Probability density p(m|L) (per methylation unit)")
    ax.set_title("Ligand-conditioned methylation states")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "conditional_methylation_distributions.png", dpi=250)
    plt.close(fig)

    # Prior sensitivity
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    centers = np.sqrt(prior_df["Lmin_uM"] * prior_df["Lmax_uM"])
    ax.plot(centers, prior_df["I_bits"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Geometric center of four-decade prior window (μM)")
    ax.set_ylabel("I(L;m) (bits)")
    ax.set_title("Environmental-prior sensitivity")
    fig.tight_layout()
    fig.savefig(OUT / "prior_window_sensitivity.png", dpi=250)
    plt.close(fig)

    # Corrected S5 ratios
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(bnd_df["g_mm^-1"], bnd_df["reflecting_over_open"], marker="o", label="Reflecting / open")
    ax.plot(bnd_df["g_mm^-1"], bnd_df["finite_window_over_open"], marker="s", label="Finite-window / open")
    ax.plot(bnd_df["g_mm^-1"], bnd_df["reinjection_over_open_stress_test"], marker="^",
            linestyle="--", label="Reinjection / open (stress test)")
    ax.axhline(1.0, linestyle=":", label="Open-domain reference")
    ax.set_xlabel(r"Gradient steepness, $g$ (mm$^{-1}$)")
    ax.set_ylabel("Rate ratio relative to open-domain model")
    ax.set_title("Boundary-condition sensitivity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_S5_boundary_ratios.png", dpi=250)
    plt.close(fig)

    # Timescale hierarchy
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(times_df["g_mm^-1"], times_df["tau_env_min"], marker="o")
    ax.set_xlabel(r"Gradient steepness, $g$ (mm$^{-1}$)")
    ax.set_ylabel(r"$\tau_{\rm env}$ (min)")
    ax.set_title("Time to traverse one information-equivalent adaptive interval")
    fig.tight_layout()
    fig.savefig(OUT / "tau_env_vs_gradient.png", dpi=250)
    plt.close(fig)

    # Dynamic rate scales
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(dyn_df["g_mm^-1"], dyn_df["methylation_open_median_bits_s"],
            marker="o", label="Methylation background (open-domain)")
    ax.fill_between(
        dyn_df["g_mm^-1"],
        dyn_df["methylation_open_lo_bits_s"],
        dyn_df["methylation_open_hi_bits_s"],
        alpha=0.2,
    )
    ax.plot(dyn_df["g_mm^-1"], dyn_df["kinase_measured_bits_s"],
            marker="s", label="Kinase benchmark")
    ax.set_yscale("log")
    ax.set_xlabel(r"Gradient steepness, $g$ (mm$^{-1}$)")
    ax.set_ylabel("Information rate (bits s⁻¹)")
    ax.set_title("Dynamic information-rate scales")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_rate_scales.png", dpi=250)
    plt.close(fig)


# =====================================
# 7. Validation
# =====================================
def validate(summary):
    expected = {
        "alpha": 1.9946442705633443,
        "m0": 0.4365445164420451,
        "GmA": 2.659525694084459,
        "I_bits": 2.181632308931791,
        "N_eff": 4.536665559535445,
    }
    tolerances = {
        "alpha": 5e-6,
        "m0": 5e-6,
        "GmA": 5e-6,
        "I_bits": 5e-4,
        "N_eff": 5e-3,
    }
    for key, target in expected.items():
        value = summary[key]
        if abs(value - target) > tolerances[key]:
            raise AssertionError(f"{key}: got {value}, expected {target} ± {tolerances[key]}")


# =====================================
# 8. Main execution
# =====================================
def main():
    H_micro, H_count = formal_capacities()

    states, scale = load_shimizu_source()
    fm_df = fit_all_fm(states, scale)
    cal = endpoint_calibration(fm_df)
    alpha = float(cal["alpha"])
    m0 = float(cal["m0"])
    GmA = float(cal["GmA"])

    within, structural = bootstrap_calibration(states, scale, fm_df, cal, n=10000)

    mstar, kappa, sigma_A, sigma_R, sigma_m = make_static_functions(alpha, m0, GmA)

    I_main, N_eff = mutual_information(alpha, m0, GmA, *PRIOR_MAIN_uM)

    prior_rows = []
    for lo, hi in PRIOR_WINDOWS_uM:
        I, N = mutual_information(alpha, m0, GmA, lo, hi)
        prior_rows.append({
            "Lmin_uM": lo,
            "Lmax_uM": hi,
            "prior_center_uM": math.sqrt(lo * hi),
            "I_bits": I,
            "N_eff": N,
        })
    prior_df = pd.DataFrame(prior_rows)

    noise_df = pd.DataFrame({
        "condition": ["buffer_displayed_at_1uM", "10uM", "25uM"],
        "L_display_uM": NOISE_L_ANCHORS_uM,
        "sigma_A_effective": [
            COLIN_SIGMA_A["buffer"], COLIN_SIGMA_A["10uM"], COLIN_SIGMA_A["25uM"]
        ],
        "sigma_R": sigma_R(NOISE_L_ANCHORS_uM),
        "sigma_m": sigma_m(NOISE_L_ANCHORS_uM),
    })

    run_df = run_scale_changes()
    times_df = timescale_table(I_main)
    dyn_df = dynamic_summary()
    bnd_df = boundary_ratios()

    # Export tables
    fm_df.to_csv(OUT / "shimizu_fitted_free_energies.csv", index=False)
    noise_df.to_csv(OUT / "colin_noise_mapping.csv", index=False)
    prior_df.to_csv(OUT / "prior_sensitivity.csv", index=False)
    run_df.to_csv(OUT / "run_scale_changes.csv", index=False)
    times_df.to_csv(OUT / "timescale_hierarchy.csv", index=False)
    dyn_df.to_csv(OUT / "dynamic_rate_summary.csv", index=False)
    bnd_df.to_csv(OUT / "boundary_ratios_Table_S3.csv", index=False)

    bootstrap_summary = pd.DataFrame([
        {
            "layer": "within-state residual bootstrap",
            "alpha_median": np.median(within[:, 0]),
            "alpha_2.5%": np.percentile(within[:, 0], 2.5),
            "alpha_97.5%": np.percentile(within[:, 0], 97.5),
            "GmA_2.5%": (4.0/3.0) * np.percentile(within[:, 0], 2.5),
            "GmA_97.5%": (4.0/3.0) * np.percentile(within[:, 0], 97.5),
        },
        {
            "layer": "between-family structural sensitivity",
            "alpha_median": np.median(structural),
            "alpha_2.5%": np.percentile(structural, 2.5),
            "alpha_97.5%": np.percentile(structural, 97.5),
            "GmA_2.5%": (4.0/3.0) * np.percentile(structural, 2.5),
            "GmA_97.5%": (4.0/3.0) * np.percentile(structural, 97.5),
        },
    ])
    bootstrap_summary.to_csv(OUT / "shimizu_uncertainty_layers.csv", index=False)

    make_figures(alpha, m0, GmA, cal, prior_df, bnd_df, times_df, dyn_df)

    summary = {
        "H_micro_bits_per_cell": H_micro,
        "H_count_bits_per_cell": H_count,
        "alpha": alpha,
        "m0": m0,
        "GmA": GmA,
        "alpha_within_state_95": [
            float(np.percentile(within[:, 0], 2.5)),
            float(np.percentile(within[:, 0], 97.5)),
        ],
        "alpha_structural_95": [
            float(np.percentile(structural, 2.5)),
            float(np.percentile(structural, 97.5)),
        ],
        "sigma_m_anchors": noise_df["sigma_m"].tolist(),
        "I_bits": I_main,
        "N_eff": N_eff,
        "I_prior_sensitivity_min": float(prior_df["I_bits"].min()),
        "I_prior_sensitivity_max": float(prior_df["I_bits"].max()),
        "N_eff_prior_sensitivity_min": float(prior_df["N_eff"].min()),
        "N_eff_prior_sensitivity_max": float(prior_df["N_eff"].max()),
        "tau_env_min_range_min": [
            float(times_df["tau_env_min"].min()),
            float(times_df["tau_env_min"].max()),
        ],
        "epsilon_ad_range": [
            float(times_df["epsilon_ad"].min()),
            float(times_df["epsilon_ad"].max()),
        ],
        "corner_frequency_Hz": 1.0 / (2.0 * math.pi * TAU_M_S),
        "dynamic_open_rates_bits_s": OPEN_RATE.tolist(),
        "kinase_rates_bits_s": dyn_df["kinase_measured_bits_s"].tolist(),
        "rate_scale_ratio_percent": dyn_df["rate_scale_ratio_percent"].tolist(),
        "boundary_reflecting_over_open": bnd_df["reflecting_over_open"].tolist(),
        "boundary_finite_window_over_open": bnd_df["finite_window_over_open"].tolist(),
    }

    validate(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== PRINCIPAL REPRODUCIBLE RESULTS ===")
    print(json.dumps(summary, indent=2))
    print("\nCSV tables and PNG figures written to:", OUT)
    print("\nIMPORTANT: final dynamic open-domain rates are frozen manuscript outputs,")
    print("not regenerated from raw trajectories in this package. See module docstring.")

    return summary


if __name__ == "__main__":
    main()
