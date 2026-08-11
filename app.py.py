"""
Microwave / Laser Metal Sintering & Melt-Pool Design Tool
---------------------------------------------------------
Streamlit engineering model for comparing a moving laser heat source with a
focused microwave heat source for powder-bed metal processing.

The app is intentionally a reduced-order design / optimization tool, not a
replacement for coupled EM-thermal-fluid simulation, calibration coupons, or
machine qualification.

Run:
    pip install -r requirements_mw_laser.txt
    streamlit run microwave_laser_sintering_app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution, minimize


# -----------------------------
# Constants
# -----------------------------
EPS0 = 8.8541878128e-12  # F/m
MU0 = 4.0e-7 * math.pi   # H/m
C0 = 299_792_458.0       # m/s
Z0 = math.sqrt(MU0 / EPS0)
R_GAS = 8.314462618  # J/mol/K


# -----------------------------
# Material presets
# -----------------------------
# Representative room/high-temperature screening values. They are deliberately
# editable because AM powder properties, absorptivity, emissivity, electrical
# connectivity, and phase-change data vary strongly with chemistry, morphology,
# temperature, packing, and machine state.
MATERIALS: Dict[str, Dict[str, float]] = {
    "AlSi10Mg (illustrative)": {
        "rho": 2670.0,
        "cp": 900.0,
        "k": 120.0,
        "Tm_C": 585.0,
        "Lf": 390_000.0,
        "alpha_exp": 22.0e-6,
        "E": 70e9,
        "nu": 0.33,
        "yield_strength": 230e6,
        "sigma_e": 2.0e7,
        "mu_r": 1.0,
        "laser_abs": 0.35,
        "eps_r": 3.0,
        "eps_loss": 0.15,
    },
    "316L stainless steel (illustrative)": {
        "rho": 8000.0,
        "cp": 500.0,
        "k": 16.0,
        "Tm_C": 1400.0,
        "Lf": 260_000.0,
        "alpha_exp": 16.0e-6,
        "E": 193e9,
        "nu": 0.30,
        "yield_strength": 290e6,
        "sigma_e": 1.35e6,
        "mu_r": 1.0,
        "laser_abs": 0.45,
        "eps_r": 5.0,
        "eps_loss": 0.30,
    },
    "Ti-6Al-4V (illustrative)": {
        "rho": 4430.0,
        "cp": 560.0,
        "k": 7.0,
        "Tm_C": 1660.0,
        "Lf": 300_000.0,
        "alpha_exp": 9.0e-6,
        "E": 114e9,
        "nu": 0.34,
        "yield_strength": 880e6,
        "sigma_e": 5.8e5,
        "mu_r": 1.0,
        "laser_abs": 0.40,
        "eps_r": 5.0,
        "eps_loss": 0.25,
    },
    "IN718 (illustrative)": {
        "rho": 8190.0,
        "cp": 435.0,
        "k": 11.4,
        "Tm_C": 1335.0,
        "Lf": 270_000.0,
        "alpha_exp": 13.0e-6,
        "E": 200e9,
        "nu": 0.29,
        "yield_strength": 1030e6,
        "sigma_e": 8.0e5,
        "mu_r": 1.0,
        "laser_abs": 0.45,
        "eps_r": 5.0,
        "eps_loss": 0.25,
    },
    "Copper (illustrative)": {
        "rho": 8960.0,
        "cp": 385.0,
        "k": 390.0,
        "Tm_C": 1084.6,
        "Lf": 205_000.0,
        "alpha_exp": 16.5e-6,
        "E": 110e9,
        "nu": 0.34,
        "yield_strength": 70e6,
        "sigma_e": 5.8e7,
        "mu_r": 1.0,
        "laser_abs": 0.20,
        "eps_r": 2.5,
        "eps_loss": 0.10,
    },
}


@dataclass
class Material:
    rho: float
    cp: float
    k: float
    Tm_K: float
    Lf: float
    alpha_exp: float
    E: float
    nu: float
    yield_strength: float
    sigma_e: float
    mu_r: float
    laser_abs: float
    eps_r: float
    eps_loss: float
    packing_fraction: float

    @property
    def rho_bed(self) -> float:
        return self.rho * self.packing_fraction

    @property
    def thermal_diffusivity_solid(self) -> float:
        return self.k / max(self.rho * self.cp, 1e-30)

    @property
    def thermal_diffusivity_bed(self) -> float:
        # Simple screening approximation: density reduction is included but
        # conductivity is not separately powder-corrected unless the user edits k.
        return self.k / max(self.rho_bed * self.cp, 1e-30)


# -----------------------------
# Physics helpers
# -----------------------------
def target_enthalpy_per_kg(mat: Material, T0_K: float, target_mode: str, sinter_fraction: float) -> Tuple[float, float]:
    """Return target temperature and specific energy (J/kg)."""
    if target_mode == "Full melting":
        T_target = mat.Tm_K
        h = mat.cp * max(T_target - T0_K, 0.0) + mat.Lf
    else:
        T_target = T0_K + sinter_fraction * max(mat.Tm_K - T0_K, 0.0)
        h = mat.cp * max(T_target - T0_K, 0.0)
    return T_target, max(h, 1.0)


def temperature_from_specific_energy(e_spec: float, mat: Material, T0_K: float) -> Tuple[float, float]:
    """Adiabatic piecewise sensible/latent temperature estimate.

    Returns (temperature K, melt_fraction 0..1).
    """
    sensible_to_melt = mat.cp * max(mat.Tm_K - T0_K, 0.0)
    if e_spec <= sensible_to_melt:
        return T0_K + e_spec / max(mat.cp, 1e-12), 0.0
    remaining = e_spec - sensible_to_melt
    if remaining <= mat.Lf:
        return mat.Tm_K, remaining / max(mat.Lf, 1e-12)
    return mat.Tm_K + (remaining - mat.Lf) / max(mat.cp, 1e-12), 1.0


def paraboloid_volume(length_m: float, width_m: float, depth_m: float) -> float:
    """Elliptic paraboloid: V = 1/2*pi*a*b*d = pi*L*W*d/8."""
    return math.pi * length_m * width_m * depth_m / 8.0


def gaussian_flux_peak(P_W: float, absorptivity: float, radius_m: float) -> float:
    return 2.0 * absorptivity * P_W / (math.pi * max(radius_m, 1e-12) ** 2)


def skin_depth(f_Hz: float, mu_r: float, sigma_Sm: float) -> float:
    # delta = sqrt(1 / (pi f mu sigma)) = sqrt(2 / (omega mu sigma))
    return math.sqrt(1.0 / max(math.pi * f_Hz * MU0 * mu_r * sigma_Sm, 1e-30))


def dielectric_penetration_depth_power(f_Hz: float, eps_r: float, eps_loss: float) -> float:
    """Power 1/e penetration depth for a nonmagnetic lossy dielectric.

    Based on the complex propagation constant. For strong conductors this is
    not a powder-bed penetration model; use skin depth and/or calibrated bed depth.
    """
    eps_r = max(eps_r, 1e-12)
    eps_loss = max(eps_loss, 1e-12)
    tan_delta = eps_loss / eps_r
    k0 = 2.0 * math.pi * f_Hz / C0
    alpha_amp = k0 * math.sqrt(eps_r / 2.0) * math.sqrt(max(math.sqrt(1.0 + tan_delta**2) - 1.0, 1e-30))
    return 1.0 / max(2.0 * alpha_amp, 1e-30)  # power decays as exp(-2 alpha z)


def thermal_stress_proxy(mat: Material, T_ref_K: float, T_hot_K: float, restraint: float) -> Tuple[float, float]:
    delta_T = max(min(T_hot_K, mat.Tm_K) - T_ref_K, 0.0)
    free_strain = mat.alpha_exp * delta_T
    stress = restraint * mat.E * free_strain / max(1.0 - mat.nu, 1e-6)
    return free_strain, stress


def pool_geometry_from_energy(
    absorbed_power_W: float,
    scan_speed_m_s: float,
    radius_m: float,
    alpha_th: float,
    H_target_J_m3: float,
    melt_efficiency: float,
) -> Dict[str, float]:
    """Reduced-order moving-source melt/heated-zone geometry.

    1) dwell time ~ 2r/v
    2) target processed volume = eta_m * P_abs * dwell / H_target
    3) diffusion broadens the transverse and longitudinal characteristic scales
    4) depth is solved so an elliptic paraboloid exactly contains that energy volume

    This is an engineering surrogate, not a CFD free-surface solution.
    """
    v = max(scan_speed_m_s, 1e-9)
    r = max(radius_m, 1e-9)
    dwell = 2.0 * r / v
    diff2 = 4.0 * max(alpha_th, 1e-16) * dwell
    width = 2.0 * math.sqrt(r**2 + diff2)
    length = 2.0 * math.sqrt(r**2 + (0.5 * v * dwell) ** 2 + diff2)
    V_energy = max(melt_efficiency, 0.0) * max(absorbed_power_W, 0.0) * dwell / max(H_target_J_m3, 1.0)
    depth = 8.0 * V_energy / max(math.pi * length * width, 1e-30)
    V_geom = paraboloid_volume(length, width, depth)
    return {
        "dwell_s": dwell,
        "length_m": length,
        "width_m": width,
        "depth_m": depth,
        "volume_m3": V_geom,
        "energy_volume_m3": V_energy,
    }


def build_time_s(build_L_m: float, build_W_m: float, build_H_m: float, v_m_s: float, hatch_m: float, layer_m: float, recoat_s: float) -> float:
    n_layers = build_H_m / max(layer_m, 1e-9)
    scan_length_per_layer = build_L_m * build_W_m / max(hatch_m, 1e-9)
    scan_time = scan_length_per_layer / max(v_m_s, 1e-9) * n_layers
    return scan_time + max(recoat_s, 0.0) * n_layers


def laser_model(
    mat: Material,
    P_W: float,
    v_mm_s: float,
    hatch_mm: float,
    layer_mm: float,
    beam_radius_um: float,
    optical_abs_depth_um: float,
    T0_C: float,
    target_mode: str,
    sinter_fraction: float,
    melt_efficiency: float,
    restraint: float,
) -> Dict[str, float]:
    v = v_mm_s / 1000.0
    h = hatch_mm / 1000.0
    t = layer_mm / 1000.0
    r = beam_radius_um * 1e-6
    abs_depth = max(optical_abs_depth_um * 1e-6, 1e-9)
    T0_K = T0_C + 273.15
    T_target, h_target_kg = target_enthalpy_per_kg(mat, T0_K, target_mode, sinter_fraction)
    H_target = mat.rho_bed * h_target_kg
    P_abs = mat.laser_abs * P_W

    pool = pool_geometry_from_energy(P_abs, v, r, mat.thermal_diffusivity_bed, H_target, melt_efficiency)

    q0 = gaussian_flux_peak(P_W, mat.laser_abs, r)
    alpha_opt = 1.0 / abs_depth
    qv0 = alpha_opt * q0
    # The reduced-order pool is *defined* as the volume brought to the selected
    # thermal target by the available energy. Therefore the zone-average thermal
    # state is the target state; a physical centerline peak temperature requires
    # a transient field solver with losses, phase change, flow and evaporation.
    T_peak_K = T_target
    melt_fraction = 1.0 if target_mode == "Full melting" else 0.0

    # Traditional bookkeeping metric and a dimensionless enthalpy scaling.
    ved_J_mm3 = P_abs / max(v_mm_s * hatch_mm * layer_mm, 1e-30)
    alpha = mat.thermal_diffusivity_solid
    deltaT = max(mat.Tm_K - T0_K, 1.0)
    ne_denom = mat.rho * mat.cp * deltaT * math.sqrt(max(math.pi * alpha * v * r**3, 1e-30))
    normalized_enthalpy = P_abs / max(ne_denom, 1e-30)

    dwell_to_diff = pool["dwell_s"] / max(r**2 / max(alpha, 1e-30), 1e-30)
    peclet = v * r / max(2.0 * alpha, 1e-30)

    # Dimensionally consistent form of the earlier width scaling.
    legacy_width = 2.0 * math.sqrt(P_abs / max(mat.rho_bed * mat.cp * deltaT * v, 1e-30))

    overlap = 1.0 - h / max(pool["width_m"], 1e-12)
    free_strain, stress = thermal_stress_proxy(mat, T0_K, T_peak_K, restraint)

    return {
        **pool,
        "P_abs_W": P_abs,
        "q_peak_W_m2": q0,
        "qv_peak_W_m3": qv0,
        "T_target_K": T_target,
        "T_peak_K": T_peak_K,
        "melt_fraction": melt_fraction,
        "VED_J_mm3": ved_J_mm3,
        "normalized_enthalpy": normalized_enthalpy,
        "dwell_to_diffusion": dwell_to_diff,
        "peclet": peclet,
        "legacy_width_m": legacy_width,
        "overlap": overlap,
        "free_strain": free_strain,
        "stress_proxy_Pa": stress,
        "stress_ratio": stress / max(mat.yield_strength, 1.0),
        "H_target_J_m3": H_target,
        "layer_m": t,
        "hatch_m": h,
        "v_m_s": v,
        "beam_radius_m": r,
        "abs_depth_m": abs_depth,
    }


def microwave_model(
    mat: Material,
    P_W: float,
    f_GHz: float,
    v_mm_s: float,
    hatch_mm: float,
    layer_mm: float,
    spot_radius_mm: float,
    bed_depth_mm: float,
    coupling_eff: float,
    field_fill_eff: float,
    T0_C: float,
    target_mode: str,
    sinter_fraction: float,
    melt_efficiency: float,
    restraint: float,
    sigma_eff_fraction: float,
) -> Dict[str, float]:
    f = f_GHz * 1e9
    omega = 2.0 * math.pi * f
    v = v_mm_s / 1000.0
    h = hatch_mm / 1000.0
    t = layer_mm / 1000.0
    r = spot_radius_mm / 1000.0
    d_eff = max(bed_depth_mm / 1000.0, 1e-9)
    T0_K = T0_C + 273.15
    T_target, h_target_kg = target_enthalpy_per_kg(mat, T0_K, target_mode, sinter_fraction)
    H_target = mat.rho_bed * h_target_kg

    # Effective source coupling and finite-depth absorption.
    absorption_fraction = 1.0 - math.exp(-t / d_eff)
    P_coupled = coupling_eff * P_W
    P_abs = P_coupled * absorption_fraction

    # Free-space focused-field estimate; diagnostic only for a cavity/near-field system.
    area = math.pi * max(r, 1e-9) ** 2
    E_rms = math.sqrt(max(field_fill_eff * P_W * Z0 / area, 0.0))
    H_rms = E_rms / Z0
    sigma_eff = mat.sigma_e * sigma_eff_fraction
    eps_loss_eff = mat.eps_loss + sigma_eff / max(omega * EPS0, 1e-30)
    qv_electric = omega * EPS0 * eps_loss_eff * E_rms**2
    qv_magnetic = 0.0  # mu'' omitted by default; many engineering alloys are weakly magnetic at process T.
    qv_total = qv_electric + qv_magnetic

    skin = skin_depth(f, mat.mu_r, max(mat.sigma_e, 1e-12))
    diel_pd = dielectric_penetration_depth_power(f, mat.eps_r, max(mat.eps_loss, 1e-9))
    wavelength = C0 / f
    airy_radius = 0.61 * wavelength  # NA=1 far-field lower-bound indicator

    pool = pool_geometry_from_energy(P_abs, v, r, mat.thermal_diffusivity_bed, H_target, melt_efficiency)

    active_volume = area * d_eff
    qv_source_limited = P_abs / max(active_volume, 1e-30)
    # As in the laser surrogate, the predicted zone volume is the volume brought
    # to the selected target state. A true local peak temperature needs coupled
    # Maxwell + transient heat transfer with temperature-dependent properties.
    T_peak_K = T_target
    melt_fraction = 1.0 if target_mode == "Full melting" else 0.0

    overlap = 1.0 - h / max(pool["width_m"], 1e-12)
    free_strain, stress = thermal_stress_proxy(mat, T0_K, T_peak_K, restraint)
    particle_skin_ratio = np.nan

    return {
        **pool,
        "P_coupled_W": P_coupled,
        "P_abs_W": P_abs,
        "absorption_fraction": absorption_fraction,
        "E_rms_V_m": E_rms,
        "H_rms_A_m": H_rms,
        "qv_field_model_W_m3": qv_total,
        "qv_source_limited_W_m3": qv_source_limited,
        "eps_loss_eff": eps_loss_eff,
        "sigma_eff_S_m": sigma_eff,
        "skin_depth_m": skin,
        "dielectric_pd_m": diel_pd,
        "wavelength_m": wavelength,
        "airy_radius_m": airy_radius,
        "T_target_K": T_target,
        "T_peak_K": T_peak_K,
        "melt_fraction": melt_fraction,
        "overlap": overlap,
        "free_strain": free_strain,
        "stress_proxy_Pa": stress,
        "stress_ratio": stress / max(mat.yield_strength, 1.0),
        "H_target_J_m3": H_target,
        "layer_m": t,
        "hatch_m": h,
        "v_m_s": v,
        "beam_radius_m": r,
        "effective_depth_m": d_eff,
        "frequency_Hz": f,
        "particle_skin_ratio": particle_skin_ratio,
    }


def pool_surface_figure(length_m: float, width_m: float, depth_m: float) -> go.Figure:
    L = max(length_m * 1e3, 1e-6)
    W = max(width_m * 1e3, 1e-6)
    D = max(depth_m * 1e3, 1e-6)
    a, b = L / 2.0, W / 2.0
    x = np.linspace(-a, a, 90)
    y = np.linspace(-b, b, 70)
    X, Y = np.meshgrid(x, y)
    R2 = (X / a) ** 2 + (Y / b) ** 2
    Z = -D * (1.0 - R2)
    Z[R2 > 1.0] = np.nan

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, showscale=False)])
    fig.update_layout(
        title="Elliptic-paraboloid processed zone",
        scene=dict(
            xaxis_title="Scan direction x (mm)",
            yaxis_title="Transverse y (mm)",
            zaxis_title="Depth z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        height=520,
    )
    return fig


def metric_table(result: Dict[str, float], method: str) -> pd.DataFrame:
    rows = [
        ("Absorbed power", result["P_abs_W"], "W"),
        ("Dwell time", result["dwell_s"] * 1e3, "ms"),
        ("Pool / zone length", result["length_m"] * 1e3, "mm"),
        ("Pool / zone width", result["width_m"] * 1e3, "mm"),
        ("Pool / zone depth", result["depth_m"] * 1e3, "mm"),
        ("Paraboloid volume", result["volume_m3"] * 1e9, "mm³"),
        ("Zone-average target T", result["T_peak_K"] - 273.15, "°C"),
        ("Melt fraction proxy", result["melt_fraction"], "0–1"),
        ("Track overlap", 100.0 * result["overlap"], "%"),
        ("Free thermal strain", 100.0 * result["free_strain"], "%"),
        ("Elastic stress proxy / yield", result["stress_ratio"], "ratio"),
    ]
    if method == "Laser":
        rows += [
            ("Absorbed VED", result["VED_J_mm3"], "J/mm³"),
            ("Normalized enthalpy", result["normalized_enthalpy"], "–"),
            ("Péclet number", result["peclet"], "–"),
            ("Dwell / diffusion time", result["dwell_to_diffusion"], "–"),
            ("Legacy width scaling", result["legacy_width_m"] * 1e3, "mm"),
        ]
    else:
        rows += [
            ("Wavelength", result["wavelength_m"] * 1e3, "mm"),
            ("Bulk-metal skin depth", result["skin_depth_m"] * 1e6, "µm"),
            ("Dielectric power penetration", result["dielectric_pd_m"] * 1e3, "mm"),
            ("Effective bed absorption", 100.0 * result["absorption_fraction"], "%"),
            ("Estimated E_rms", result["E_rms_V_m"], "V/m"),
            ("Source-limited q'''", result["qv_source_limited_W_m3"], "W/m³"),
        ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


def symbol_table(rows):
    """Render an unmistakable variable/units list directly beneath one equation."""
    with st.container(border=True):
        st.markdown("**Variables**")
        lines = []
        for symbol, meaning, unit in rows:
            lines.append(f"- **{symbol}** — {meaning}  \n  *Units:* `{unit}`")
        st.markdown("\n".join(lines))


def feasibility(result: Dict[str, float], layer_mm: float, min_overlap: float, target_mode: str, T_target_K: float) -> Dict[str, bool]:
    return {
        "Depth ≥ layer": result["depth_m"] >= layer_mm / 1000.0,
        "Overlap ≥ target": result["overlap"] >= min_overlap,
        "Positive geometry": result["width_m"] > 0 and result["depth_m"] > 0 and result["length_m"] > 0,
    }


def pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    d = df.sort_values(x, ascending=True).copy()
    keep = []
    best_y = float("inf")
    for idx, row in d.iterrows():
        if row[y] < best_y:
            keep.append(idx)
            best_y = row[y]
    return d.loc[keep].sort_values(x)


# -----------------------------
# Streamlit layout
# -----------------------------
st.set_page_config(page_title="Microwave vs Laser Metal Sintering", page_icon="⚙️", layout="wide")

st.title("Microwave vs. Laser Metal Sintering / Melting")
st.caption(
    "Reduced-order design, equation, melt-pool, and optimization tool for metal powder processing. "
    "The melt/heated zone is represented as an elliptic paraboloid rather than a hemisphere."
)

with st.sidebar:
    st.header("1 · Process")
    method = st.radio("Energy source", ["Laser", "Microwave"], horizontal=True)
    target_mode = st.radio("Thermal target", ["Full melting", "Solid-state sintering"], horizontal=False)
    sinter_fraction = st.slider("Sinter target fraction of (Tm − T0)", 0.50, 0.98, 0.88, 0.01, disabled=(target_mode == "Full melting"))

    st.header("2 · Material")
    mat_name = st.selectbox("Preset", list(MATERIALS.keys()))
    base = MATERIALS[mat_name]

    with st.expander("Edit thermophysical properties", expanded=False):
        rho = st.number_input("Solid density ρ (kg/m³)", 100.0, 25000.0, float(base["rho"]), 10.0)
        cp = st.number_input("Specific heat cp (J/kg·K)", 50.0, 3000.0, float(base["cp"]), 10.0)
        k = st.number_input("Thermal conductivity k (W/m·K)", 0.1, 1000.0, float(base["k"]), 1.0)
        Tm_C = st.number_input("Melting / solidus temperature (°C)", 100.0, 4000.0, float(base["Tm_C"]), 10.0)
        Lf = st.number_input("Latent heat of fusion Lf (J/kg)", 0.0, 2_000_000.0, float(base["Lf"]), 5000.0)
        alpha_exp = st.number_input("Thermal expansion αL (1/K)", 1e-7, 1e-4, float(base["alpha_exp"]), format="%.3e")
        E = st.number_input("Elastic modulus E (GPa)", 0.1, 1000.0, float(base["E"] / 1e9), 1.0) * 1e9
        nu = st.number_input("Poisson ratio ν", 0.01, 0.49, float(base["nu"]), 0.01)
        yield_strength = st.number_input("Yield strength (MPa)", 1.0, 5000.0, float(base["yield_strength"] / 1e6), 10.0) * 1e6
        sigma_e = st.number_input("Bulk electrical conductivity σ (S/m)", 1e3, 1e8, float(base["sigma_e"]), format="%.3e")
        mu_r = st.number_input("Relative permeability μr", 0.1, 1000.0, float(base["mu_r"]), format="%.3f")
        laser_abs = st.number_input("Laser absorptivity A", 0.01, 0.99, float(base["laser_abs"]), 0.01)
        eps_r = st.number_input("Effective relative permittivity ε′", 1.0, 1000.0, float(base["eps_r"]), 0.1)
        eps_loss = st.number_input("Effective loss factor ε″", 1e-4, 1000.0, float(base["eps_loss"]), format="%.4f")

    packing_fraction = st.slider("Powder packing fraction", 0.20, 0.80, 0.55, 0.01)
    T0_C = st.number_input("Bed / preheat temperature (°C)", -50.0, 1500.0, 25.0, 10.0)
    melt_efficiency = st.slider("Energy-to-target-volume efficiency", 0.05, 1.0, 0.55, 0.05,
                                help="Lumps conduction, radiation, evaporation, convection and unmodeled losses into one calibration factor.")
    restraint = st.slider("Thermal restraint factor", 0.0, 1.0, 0.20, 0.05,
                          help="Used only for an elastic residual-stress screening proxy; 0 = free contraction, 1 = fully restrained.")

mat = Material(
    rho=rho, cp=cp, k=k, Tm_K=Tm_C + 273.15, Lf=Lf, alpha_exp=alpha_exp,
    E=E, nu=nu, yield_strength=yield_strength, sigma_e=sigma_e, mu_r=mu_r,
    laser_abs=laser_abs, eps_r=eps_r, eps_loss=eps_loss, packing_fraction=packing_fraction,
)

# Main process inputs
st.subheader("Process inputs")
if method == "Laser":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        P_W = st.number_input("Laser power P (W)", 1.0, 5000.0, 300.0, 10.0)
        beam_radius_um = st.number_input("1/e² beam radius rb (µm)", 5.0, 2000.0, 50.0, 5.0)
    with c2:
        v_mm_s = st.number_input("Scan speed v (mm/s)", 1.0, 10000.0, 1000.0, 50.0)
        optical_abs_depth_um = st.number_input("Optical absorption depth 1/β (µm)", 0.01, 1000.0, 20.0, 1.0)
    with c3:
        hatch_mm = st.number_input("Hatch spacing h (mm)", 0.001, 5.0, 0.10, 0.01, format="%.3f")
    with c4:
        layer_mm = st.number_input("Layer thickness t (mm)", 0.001, 2.0, 0.03, 0.005, format="%.3f")

    result = laser_model(
        mat, P_W, v_mm_s, hatch_mm, layer_mm, beam_radius_um, optical_abs_depth_um,
        T0_C, target_mode, sinter_fraction, melt_efficiency, restraint,
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        P_W = st.number_input("Microwave source power P (W)", 1.0, 100000.0, 2000.0, 100.0)
        f_GHz = st.number_input("Frequency f (GHz)", 0.1, 300.0, 2.45, 0.05)
    with c2:
        v_mm_s = st.number_input("Scan speed v (mm/s)", 0.1, 5000.0, 25.0, 5.0)
        spot_radius_mm = st.number_input("Effective focused radius r (mm)", 0.01, 1000.0, 2.0, 0.1)
    with c3:
        hatch_mm = st.number_input("Hatch spacing h (mm)", 0.001, 100.0, 2.0, 0.1, format="%.3f")
        layer_mm = st.number_input("Layer thickness t (mm)", 0.001, 20.0, 0.20, 0.05, format="%.3f")
    with c4:
        bed_depth_mm = st.number_input("Effective powder absorption depth (mm)", 0.001, 1000.0, 1.0, 0.1)
        coupling_eff = st.slider("Source → powder coupling ηc", 0.01, 1.0, 0.35, 0.01)
        field_fill_eff = st.slider("Field-fill efficiency", 0.01, 1.0, 0.30, 0.01,
                                   help="Diagnostic conversion from source power to a plane-wave-equivalent local E-field.")
        sigma_eff_fraction = st.number_input("Powder effective σ / bulk σ", 1e-10, 1.0, 1e-5, format="%.2e",
                                             help="Loose powder electrical connectivity can be orders of magnitude below bulk. Used only in the field-loss diagnostic.")

    result = microwave_model(
        mat, P_W, f_GHz, v_mm_s, hatch_mm, layer_mm, spot_radius_mm, bed_depth_mm,
        coupling_eff, field_fill_eff, T0_C, target_mode, sinter_fraction,
        melt_efficiency, restraint, sigma_eff_fraction,
    )

T_target_K = result["T_target_K"]
min_overlap = 0.10

# Summary metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Depth", f"{result['depth_m']*1e3:.3f} mm", f"layer {layer_mm:.3f} mm")
m2.metric("Width", f"{result['width_m']*1e3:.3f} mm")
m3.metric("Length", f"{result['length_m']*1e3:.3f} mm")
m4.metric("Zone target T", f"{result['T_peak_K']-273.15:.0f} °C")
m5.metric("Track overlap", f"{100*result['overlap']:.1f} %")

checks = feasibility(result, layer_mm, min_overlap, target_mode, T_target_K)
if all(checks.values()):
    st.success("Reduced-order feasibility checks pass for the current settings.")
else:
    failing = [k for k, ok in checks.items() if not ok]
    st.warning("Current screening checks not met: " + "; ".join(failing))

# Tabs
(tab_overview, tab_eq, tab_pool, tab_opt, tab_powder, tab_refs) = st.tabs([
    "Method comparison", "Equations", "Paraboloid melt pool", "Optimization", "Powder / wavelength", "References & limits"
])

with tab_overview:
    st.markdown("### What the app is solving")
    st.write(
        "Both routes are reduced to the same thermal target: deliver enough absorbed energy during the local residence time "
        "to raise a packed powder volume to a selected solid-state sintering temperature or through fusion. The predicted "
        "processed zone is then fitted to an elliptic paraboloid whose depth is determined from the energy balance."
    )

    comp = pd.DataFrame([
        ["Energy coupling", "Optical absorption; Gaussian beam", "EM loss / induced currents; source coupling and penetration"],
        ["Spatial localization", "Typically tens–hundreds of µm", "Far-field spot tied to wavelength; near-field/cavity focusing can be smaller"],
        ["Primary screening metric", "Normalized enthalpy + Péclet/diffusion ratio", "Wavelength, skin depth, effective penetration, q''' and coupling"],
        ["Main calibration need", "Absorptivity, spot profile, melt efficiency", "Effective bed penetration, powder conductivity/loss, field distribution"],
        ["Failure modes omitted here", "Keyhole, recoil pressure, Marangoni flow, spatter", "Arcing/plasma, cavity hot spots, field detuning, tooling resonance"],
    ], columns=["Feature", "Laser", "Microwave"])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("### Current result")
    dfm = metric_table(result, method)
    st.dataframe(dfm.style.format({"Value": "{:.5g}"}), use_container_width=True, hide_index=True)

    st.info(
        "The elastic thermal-stress result is intentionally a screening proxy. Once material melts, an elastic fully-solid stress model is not valid; "
        "use it to compare thermal severity, not as a final residual-stress prediction."
    )

with tab_eq:
    st.caption("Equation reference v1.2 — every displayed equation is followed immediately by a bordered Variables + Units list.")
    st.markdown("## Shared thermal target")

    st.latex(r"\alpha_{th}=\frac{k}{\rho c_p}")
    symbol_table([
        ("α_th", "Thermal diffusivity", "m²/s"),
        ("k", "Thermal conductivity", "W/(m·K)"),
        ("ρ", "Solid material density", "kg/m³"),
        ("c_p", "Specific heat capacity at constant pressure", "J/(kg·K)"),
    ])

    st.latex(r"h_m=c_p(T_m-T_0)+L_f \quad \text{(full melting)}")
    symbol_table([
        ("h_m", "Specific enthalpy required to heat and fully melt the material", "J/kg"),
        ("c_p", "Specific heat capacity", "J/(kg·K)"),
        ("T_m", "Melting / liquidus target temperature", "K"),
        ("T_0", "Initial or preheat temperature", "K"),
        ("L_f", "Latent heat of fusion", "J/kg"),
    ])

    st.latex(r"H_m=\rho_{bed} h_m, \qquad \rho_{bed}=\phi\rho")
    symbol_table([
        ("H_m", "Volumetric enthalpy required by the packed powder bed", "J/m³"),
        ("ρ_bed", "Bulk density of the packed powder bed", "kg/m³"),
        ("h_m", "Specific heating / melting enthalpy", "J/kg"),
        ("φ", "Powder packing fraction", "dimensionless (0–1)"),
        ("ρ", "Fully dense solid density", "kg/m³"),
    ])
    st.write("For solid-state sintering, the latent-heat term is omitted and the target temperature is set below the solidus/melting point.")

    st.markdown("## Moving-source energy volume")
    st.latex(r"t_d \approx \frac{2r}{v}")
    symbol_table([
        ("t_d", "Local beam/source dwell or residence time", "s"),
        ("r", "Effective source radius", "m"),
        ("v", "Source scan speed", "m/s"),
    ])

    st.latex(r"V_E=\frac{\eta_m P_{abs} t_d}{H_m}")
    symbol_table([
        ("V_E", "Energy-limited processed volume", "m³"),
        ("η_m", "Effective melt/sinter energy-to-volume efficiency", "dimensionless (0–1)"),
        ("P_abs", "Power absorbed by the powder/workpiece", "W"),
        ("t_d", "Local dwell time", "s"),
        ("H_m", "Required volumetric enthalpy", "J/m³"),
    ])

    st.latex(r"W=2\sqrt{r^2+4\alpha_{th}t_d}")
    symbol_table([
        ("W", "Predicted processed-zone width", "m"),
        ("r", "Effective source radius", "m"),
        ("α_th", "Thermal diffusivity", "m²/s"),
        ("t_d", "Local dwell time", "s"),
    ])

    st.latex(r"L=2\sqrt{r^2+(vt_d/2)^2+4\alpha_{th}t_d}")
    symbol_table([
        ("L", "Predicted processed-zone length in scan direction", "m"),
        ("r", "Effective source radius", "m"),
        ("v", "Scan speed", "m/s"),
        ("t_d", "Local dwell time", "s"),
        ("α_th", "Thermal diffusivity", "m²/s"),
    ])

    st.latex(r"V_{paraboloid}=\frac{1}{2}\pi a b d=\frac{\pi L W d}{8}")
    symbol_table([
        ("V_paraboloid", "Volume of the elliptic-paraboloid melt/heated zone", "m³"),
        ("a", "Paraboloid semi-axis in scan direction; a = L/2", "m"),
        ("b", "Paraboloid transverse semi-axis; b = W/2", "m"),
        ("d", "Maximum melt/heated-zone depth", "m"),
        ("L", "Full melt/heated-zone length", "m"),
        ("W", "Full melt/heated-zone width", "m"),
    ])

    st.latex(r"d=\frac{8V_E}{\pi L W}")
    symbol_table([
        ("d", "Energy-balanced paraboloid depth", "m"),
        ("V_E", "Energy-limited processed volume", "m³"),
        ("L", "Processed-zone length", "m"),
        ("W", "Processed-zone width", "m"),
    ])
    st.caption("These relations are the transparent reduced-order surrogate used by this app; calibrate η_m and geometry against measured tracks or a higher-fidelity solver.")

    st.markdown("## Governing heat equation / moving-source reference")
    st.latex(r"\rho c_p\frac{\partial T}{\partial t}=\nabla\cdot(k\nabla T)+\dot{Q}-\rho c_p\mathbf{v}\cdot\nabla T")
    symbol_table([
        ("ρ", "Material or effective powder-bed density", "kg/m³"),
        ("c_p", "Specific heat capacity", "J/(kg·K)"),
        ("T", "Temperature field", "K"),
        ("t", "Time", "s"),
        ("k", "Thermal conductivity", "W/(m·K)"),
        ("∇", "Spatial gradient operator", "1/m"),
        ("Q̇", "Volumetric heat-generation/source term", "W/m³"),
        ("v⃗", "Heat-source translation velocity vector", "m/s"),
    ])

    st.latex(r"T-T_0=\frac{Q}{2\pi kR}\exp\left[-\frac{v(R+\xi)}{2\alpha_{th}}\right]")
    symbol_table([
        ("T", "Temperature at the evaluation point", "K"),
        ("T_0", "Far-field / initial temperature", "K"),
        ("Q", "Net moving-point-source heat input", "W"),
        ("k", "Thermal conductivity", "W/(m·K)"),
        ("R", "Distance from the moving source to the field point", "m"),
        ("v", "Source translation speed", "m/s"),
        ("ξ", "Coordinate measured along the moving-source direction", "m"),
        ("α_th", "Thermal diffusivity", "m²/s"),
    ])
    st.caption("This is the classical 3-D Rosenthal moving-point-source solution. It is shown as a next-fidelity analytical model; the app's default pool geometry uses the faster energy/paraboloid surrogate.")

    st.markdown("## Laser model")
    st.latex(r"q''(r)=\frac{2AP}{\pi r_b^2}\exp\left(-\frac{2r^2}{r_b^2}\right)")
    symbol_table([
        ("q''(r)", "Gaussian absorbed surface heat flux at radial position r", "W/m²"),
        ("A", "Laser absorptivity", "dimensionless (0–1)"),
        ("P", "Incident laser power", "W"),
        ("r", "Radial distance from the beam center", "m"),
        ("r_b", "Gaussian beam radius used by the model", "m"),
    ])

    st.latex(r"q'''(r,z)=\beta q''(r)e^{-\beta z},\qquad \beta=1/d_{opt}")
    symbol_table([
        ("q'''(r,z)", "Volumetric absorbed laser heating rate", "W/m³"),
        ("β", "Optical attenuation coefficient", "1/m"),
        ("q''(r)", "Surface heat flux at radial position r", "W/m²"),
        ("z", "Depth below the powder-bed surface", "m"),
        ("d_opt", "Effective optical penetration depth", "m"),
    ])

    st.latex(r"VED_{abs}=\frac{AP}{vht}")
    symbol_table([
        ("VED_abs", "Absorbed volumetric energy density", "J/m³; app also reports J/mm³"),
        ("A", "Absorptivity", "dimensionless (0–1)"),
        ("P", "Laser power", "W"),
        ("v", "Scan speed", "m/s"),
        ("h", "Hatch spacing", "m"),
        ("t", "Powder layer thickness", "m"),
    ])

    st.latex(r"\frac{\Delta H}{h_s}=\frac{AP}{\rho c_p(T_m-T_0)\sqrt{\pi\alpha_{th}vr_b^3}}")
    symbol_table([
        ("ΔH/h_s", "Normalized enthalpy / dimensionless process-energy ratio", "dimensionless"),
        ("A", "Laser absorptivity", "dimensionless (0–1)"),
        ("P", "Laser power", "W"),
        ("ρ", "Material density", "kg/m³"),
        ("c_p", "Specific heat capacity", "J/(kg·K)"),
        ("T_m − T_0", "Temperature rise to melting target", "K"),
        ("α_th", "Thermal diffusivity", "m²/s"),
        ("v", "Scan speed", "m/s"),
        ("r_b", "Beam radius", "m"),
    ])

    st.latex(r"Pe=\frac{v r_b}{2\alpha_{th}}")
    symbol_table([
        ("Pe", "Péclet number; translation relative to thermal diffusion", "dimensionless"),
        ("v", "Scan speed", "m/s"),
        ("r_b", "Beam radius", "m"),
        ("α_th", "Thermal diffusivity", "m²/s"),
    ])

    st.latex(r"W_{legacy}=2\sqrt{\frac{AP}{\rho_{bed}c_p(T_m-T_0)v}}")
    symbol_table([
        ("W_legacy", "Legacy energy-balance width scaling", "m"),
        ("A", "Absorptivity", "dimensionless (0–1)"),
        ("P", "Laser power", "W"),
        ("ρ_bed", "Packed-powder bulk density", "kg/m³"),
        ("c_p", "Specific heat capacity", "J/(kg·K)"),
        ("T_m − T_0", "Temperature rise to melting target", "K"),
        ("v", "Scan speed", "m/s"),
    ])
    st.caption("The normalized-enthalpy expression follows a common LPBF scaling-law form; the beam-size convention varies by source. The legacy-width form includes ΔT so it remains dimensionally consistent.")

    st.markdown("## Microwave model")
    st.latex(r"\nabla\times(\mu_r^{-1}\nabla\times\mathbf{E})-k_0^2\left(\epsilon_r-j\frac{\sigma}{\omega\epsilon_0}\right)\mathbf{E}=0")
    symbol_table([
        ("E⃗", "Electric-field phasor", "V/m"),
        ("μ_r", "Relative magnetic permeability", "dimensionless"),
        ("∇", "Spatial differential operator", "1/m"),
        ("k_0", "Free-space wavenumber, k₀ = ω/c", "1/m"),
        ("ε_r", "Relative permittivity", "dimensionless"),
        ("j", "Imaginary unit", "dimensionless"),
        ("σ", "Electrical conductivity", "S/m"),
        ("ω", "Angular frequency, 2πf", "rad/s"),
        ("ε_0", "Vacuum permittivity", "F/m"),
    ])

    st.latex(r"\lambda_0=\frac{c}{f}")
    symbol_table([
        ("λ_0", "Free-space microwave wavelength", "m"),
        ("c", "Speed of light in vacuum", "m/s"),
        ("f", "Microwave frequency", "Hz = 1/s"),
    ])

    st.latex(r"\delta=\sqrt{\frac{1}{\pi f\mu\sigma}}=\sqrt{\frac{2}{\omega\mu\sigma}}")
    symbol_table([
        ("δ", "Bulk-conductor electromagnetic skin depth", "m"),
        ("f", "Microwave frequency", "Hz"),
        ("μ", "Absolute magnetic permeability", "H/m"),
        ("σ", "Electrical conductivity", "S/m"),
        ("ω", "Angular frequency, 2πf", "rad/s"),
    ])

    st.latex(r"q'''=\omega\epsilon_0\epsilon''_{eff}E_{rms}^2+\omega\mu_0\mu''_{eff}H_{rms}^2")
    symbol_table([
        ("q'''", "Volumetric microwave heat-generation rate", "W/m³"),
        ("ω", "Angular frequency", "rad/s"),
        ("ε_0", "Vacuum permittivity", "F/m"),
        ("ε''_eff", "Effective dielectric loss factor", "dimensionless"),
        ("E_rms", "RMS electric-field magnitude", "V/m"),
        ("μ_0", "Vacuum permeability", "H/m"),
        ("μ''_eff", "Effective magnetic loss factor", "dimensionless"),
        ("H_rms", "RMS magnetic-field magnitude", "A/m"),
    ])

    st.latex(r"\epsilon''_{eff}=\epsilon''+\frac{\sigma_{eff}}{\omega\epsilon_0}")
    symbol_table([
        ("ε''_eff", "Combined effective dielectric loss factor", "dimensionless"),
        ("ε''", "Intrinsic dielectric loss factor", "dimensionless"),
        ("σ_eff", "Effective electrical conductivity of the powder bed", "S/m"),
        ("ω", "Angular frequency", "rad/s"),
        ("ε_0", "Vacuum permittivity", "F/m"),
    ])

    st.latex(r"\frac{dT}{dt}\approx\frac{q'''}{\rho_{bed}c_p}")
    symbol_table([
        ("dT/dt", "Local temperature-rise rate", "K/s"),
        ("q'''", "Volumetric microwave heating rate", "W/m³"),
        ("ρ_bed", "Powder-bed bulk density", "kg/m³"),
        ("c_p", "Specific heat capacity", "J/(kg·K)"),
    ])

    st.latex(r"P_{abs}=\eta_c P\left(1-e^{-\ell_{bed}/d_p}\right)")
    symbol_table([
        ("P_abs", "Microwave power absorbed within the modeled bed thickness", "W"),
        ("η_c", "Source-to-bed coupling efficiency", "dimensionless (0–1)"),
        ("P", "Incident/source microwave power", "W"),
        ("ℓ_bed", "Effective powder-bed interaction thickness", "m"),
        ("d_p", "Effective microwave power-absorption / penetration depth in powder", "m"),
    ])
    st.warning(
        "For a conductive metal powder bed, bulk-metal skin depth is not the same thing as bed-scale microwave penetration. "
        "Particle contacts, oxides, packing, temperature, resonant tooling, magnetic loss and plasma/arcing can dominate. "
        "The app therefore keeps an editable effective powder absorption depth."
    )

    st.markdown("## Solid-state sintering kinetics (generic screening form)")
    st.latex(r"D(T)=D_0\exp\left(-\frac{Q_a}{RT}\right)")
    symbol_table([
        ("D(T)", "Temperature-dependent diffusion coefficient", "m²/s"),
        ("D_0", "Diffusion pre-exponential factor", "m²/s"),
        ("Q_a", "Activation energy for the selected diffusion mechanism", "J/mol"),
        ("R", "Universal gas constant", "J/(mol·K)"),
        ("T", "Absolute temperature", "K"),
    ])

    st.latex(r"k_s(T)=A_0\exp\left(-\frac{Q_a}{RT}\right),\qquad \Theta_s=\int k_s(T)\,dt")
    symbol_table([
        ("k_s(T)", "Temperature-dependent sintering rate proxy", "1/s"),
        ("A_0", "Sintering-rate pre-exponential factor", "1/s"),
        ("Q_a", "Effective activation energy", "J/mol"),
        ("R", "Universal gas constant", "J/(mol·K)"),
        ("T", "Absolute temperature", "K"),
        ("Θ_s", "Integrated Arrhenius thermal dose", "dimensionless for k_s in 1/s"),
        ("t", "Time", "s"),
    ])
    st.caption("Diffusion path, particle size, surface curvature, pressure and neck-growth mechanism determine the actual densification law. The app uses an optional Arrhenius thermal-dose diagnostic rather than claiming one universal neck-growth exponent.")

    st.markdown("## Optimization formulation")
    st.latex(r"t_{build}\approx\frac{XY}{vh}\frac{Z}{t}+N_{layers}t_{recoat}")
    symbol_table([
        ("t_build", "Estimated total raster build time", "s"),
        ("X, Y", "In-plane build dimensions", "m"),
        ("Z", "Total build height", "m"),
        ("v", "Scan speed", "m/s"),
        ("h", "Hatch spacing", "m"),
        ("t", "Layer thickness", "m"),
        ("N_layers", "Number of deposited layers", "dimensionless count"),
        ("t_recoat", "Recoating / layer-change time per layer", "s"),
    ])

    st.latex(r"\min_{P,v,h,t}\;t_{build}+\lambda\sum_i\max(0,g_i)^2")
    symbol_table([
        ("P", "Source power decision variable", "W"),
        ("v", "Scan-speed decision variable", "m/s"),
        ("h", "Hatch-spacing decision variable", "m"),
        ("t", "Layer-thickness decision variable", "m"),
        ("t_build", "Estimated build-time objective", "s"),
        ("λ", "Penalty weight; units depend on normalization of g_i", "typically s if g_i is dimensionless"),
        ("g_i", "Normalized constraint-violation function; feasible when g_i ≤ 0", "dimensionless when normalized"),
        ("i", "Constraint index", "dimensionless index"),
    ])
    st.write("Active constraints include minimum penetration depth, minimum track overlap, variable bounds, and optionally a normalized-enthalpy window. Differential Evolution supplies a global search; SLSQP locally refines the best candidate. A separate random feasible sweep estimates a time-versus-energy Pareto front.")

    st.markdown("## Thermal strain / stress screening")
    st.latex(r"\epsilon_{th}=\alpha_L\Delta T")
    symbol_table([
        ("ε_th", "Free thermal strain", "dimensionless; often reported as %"),
        ("α_L", "Linear coefficient of thermal expansion", "1/K"),
        ("ΔT", "Temperature change from reference state", "K"),
    ])

    st.latex(r"\sigma_{th,proxy}=R_s\frac{E_Y\alpha_L\Delta T}{1-\nu}")
    symbol_table([
        ("σ_th,proxy", "Elastic thermal-stress screening proxy", "Pa"),
        ("R_s", "User-set restraint factor", "dimensionless (0–1)"),
        ("E_Y", "Young's modulus", "Pa"),
        ("α_L", "Linear coefficient of thermal expansion", "1/K"),
        ("ΔT", "Temperature change", "K"),
        ("ν", "Poisson's ratio", "dimensionless"),
    ])
    st.caption("R_s is the user-set restraint factor. This is not a solidification residual-stress simulation. The subscripted notation avoids confusion with R used for Rosenthal distance and the universal gas constant in the Arrhenius equations.")

with tab_pool:
    st.plotly_chart(pool_surface_figure(result["length_m"], result["width_m"], result["depth_m"]), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"V=\frac{1}{2}\pi a b d")
        st.write(f"a = L/2 = {result['length_m']*500:.4f} mm")
        st.write(f"b = W/2 = {result['width_m']*500:.4f} mm")
        st.write(f"d = {result['depth_m']*1e3:.4f} mm")
    with c2:
        st.write("Why a paraboloid is preferable to a hemisphere in this reduced-order model:")
        st.markdown(
            "- independent **length**, **width**, and **depth** for a moving source;\n"
            "- naturally represents an elongated scan-direction footprint;\n"
            "- lets the energy balance solve depth while diffusion sets lateral scales;\n"
            "- provides a simple closed-form volume for optimization."
        )
        st.caption("Real LPBF pools can be asymmetric and become keyhole-like; a paraboloid should not be used to represent a deep vapor cavity without calibration.")

with tab_opt:
    st.markdown("## Constrained parameter optimization")
    st.write(
        "The optimizer minimizes estimated raster build time, with penalty terms enforcing layer penetration, track overlap, and the selected thermal target. "
        "A global Differential Evolution search can be followed by local SLSQP refinement."
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        build_L_mm = st.number_input("Build X (mm)", 1.0, 5000.0, 50.0, 5.0, key="buildL")
    with b2:
        build_W_mm = st.number_input("Build Y (mm)", 1.0, 5000.0, 50.0, 5.0, key="buildW")
    with b3:
        build_H_mm = st.number_input("Build Z (mm)", 0.1, 5000.0, 10.0, 1.0, key="buildH")
    with b4:
        recoat_s = st.number_input("Recoater / layer overhead (s)", 0.0, 120.0, 2.0, 0.5)

    min_overlap_opt = st.slider("Minimum track overlap", -0.25, 0.80, 0.10, 0.05)
    depth_margin = st.slider("Required depth / layer", 0.50, 5.0, 1.10, 0.10)
    penalty_weight = st.number_input("Constraint penalty weight", 1e2, 1e10, 1e6, format="%.1e")

    st.markdown("### Decision-variable bounds")
    if method == "Laser":
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            P_lo, P_hi = st.slider("Power bounds (W)", 1.0, 5000.0, (100.0, 800.0), 10.0)
        with q2:
            v_lo, v_hi = st.slider("Speed bounds (mm/s)", 1.0, 10000.0, (200.0, 2500.0), 50.0)
        with q3:
            h_lo, h_hi = st.slider("Hatch bounds (mm)", 0.001, 2.0, (0.05, 0.25), 0.005)
        with q4:
            t_lo, t_hi = st.slider("Layer bounds (mm)", 0.001, 0.5, (0.02, 0.08), 0.005)
    else:
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            P_lo, P_hi = st.slider("Power bounds (W)", 1.0, 100000.0, (500.0, 10000.0), 100.0)
        with q2:
            v_lo, v_hi = st.slider("Speed bounds (mm/s)", 0.1, 5000.0, (1.0, 200.0), 1.0)
        with q3:
            h_lo, h_hi = st.slider("Hatch bounds (mm)", 0.001, 50.0, (0.25, 5.0), 0.05)
        with q4:
            t_lo, t_hi = st.slider("Layer bounds (mm)", 0.001, 10.0, (0.05, 1.0), 0.01)

    use_ne_window = False
    ne_lo, ne_hi = 0.0, 1e9
    if method == "Laser":
        use_ne_window = st.checkbox(
            "Constrain normalized enthalpy to a user-defined window",
            value=False,
            help="Do not treat one literature threshold as universal across alloys, optics, powder states, and machines."
        )
        if use_ne_window:
            n1, n2 = st.columns(2)
            ne_lo = n1.number_input("NE lower bound", 0.0, 1000.0, 5.0, 0.5)
            ne_hi = n2.number_input("NE upper bound", 0.0, 1000.0, 10.0, 0.5)

    def evaluate_candidate(x: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        P, vv, hh, tt = [float(z) for z in x]
        if method == "Laser":
            rr = laser_model(
                mat, P, vv, hh, tt, beam_radius_um, optical_abs_depth_um,
                T0_C, target_mode, sinter_fraction, melt_efficiency, restraint,
            )
        else:
            rr = microwave_model(
                mat, P, f_GHz, vv, hh, tt, spot_radius_mm, bed_depth_mm,
                coupling_eff, field_fill_eff, T0_C, target_mode, sinter_fraction,
                melt_efficiency, restraint, sigma_eff_fraction,
            )

        tbuild = build_time_s(
            build_L_mm / 1000.0, build_W_mm / 1000.0, build_H_mm / 1000.0,
            vv / 1000.0, hh / 1000.0, tt / 1000.0, recoat_s,
        )
        p_depth = max(0.0, depth_margin * tt / 1000.0 - rr["depth_m"]) / max(tt / 1000.0, 1e-9)
        p_overlap = max(0.0, min_overlap_opt - rr["overlap"])
        p_temp = 0.0  # thermal target is built into the energy-defined pool volume
        p_ne = 0.0
        if method == "Laser" and use_ne_window:
            ne = rr["normalized_enthalpy"]
            if ne < ne_lo:
                p_ne = (ne_lo - ne) / max(ne_lo, 1.0)
            elif ne > ne_hi:
                p_ne = (ne - ne_hi) / max(ne_hi, 1.0)

        penalty = penalty_weight * (p_depth**2 + p_overlap**2 + p_temp**2 + p_ne**2)
        obj = tbuild + penalty
        diagnostics = {
            "build_time_s": tbuild,
            "penalty": penalty,
            "p_depth": p_depth,
            "p_overlap": p_overlap,
            "p_temp": p_temp,
            "p_ne": p_ne,
        }
        return obj, rr, diagnostics

    def objective(x: np.ndarray) -> float:
        return evaluate_candidate(x)[0]

    bounds = [(P_lo, P_hi), (v_lo, v_hi), (h_lo, h_hi), (t_lo, t_hi)]

    if st.button("Run global + local optimization", type="primary"):
        with st.spinner("Optimizing reduced-order process window…"):
            de = differential_evolution(objective, bounds=bounds, seed=7, maxiter=45, popsize=8, polish=False, workers=1)
            local = minimize(objective, de.x, method="SLSQP", bounds=bounds, options={"maxiter": 250, "ftol": 1e-8})
            xbest = local.x if local.fun <= de.fun else de.x
            fbest, rbest, dbest = evaluate_candidate(xbest)

        st.session_state["opt_result"] = {
            "x": xbest.tolist(), "f": float(fbest), "result": rbest, "diag": dbest,
            "method": method,
        }

    if "opt_result" in st.session_state and st.session_state["opt_result"].get("method") == method:
        opt = st.session_state["opt_result"]
        P_b, v_b, h_b, t_b = opt["x"]
        rb = opt["result"]
        db = opt["diag"]
        st.markdown("### Best candidate")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Power", f"{P_b:.2f} W")
        o2.metric("Scan speed", f"{v_b:.2f} mm/s")
        o3.metric("Hatch", f"{h_b:.4f} mm")
        o4.metric("Layer", f"{t_b:.4f} mm")
        o5, o6, o7, o8 = st.columns(4)
        o5.metric("Build time", f"{db['build_time_s']/60:.2f} min")
        o6.metric("Predicted depth", f"{rb['depth_m']*1e3:.4f} mm")
        o7.metric("Overlap", f"{100*rb['overlap']:.1f}%")
        o8.metric("Zone target T", f"{rb['T_peak_K']-273.15:.0f} °C")
        if db["penalty"] > 1e-6:
            st.warning("The best point still carries a constraint penalty. Expand bounds, reduce required overlap/depth margin, or recalibrate the physics parameters.")
        else:
            st.success("The best point meets the active reduced-order constraints within numerical tolerance.")

        export = pd.DataFrame([{
            "method": method, "power_W": P_b, "speed_mm_s": v_b, "hatch_mm": h_b, "layer_mm": t_b,
            "build_time_s": db["build_time_s"], "pool_length_mm": rb["length_m"]*1e3,
            "pool_width_mm": rb["width_m"]*1e3, "pool_depth_mm": rb["depth_m"]*1e3,
            "peak_T_C": rb["T_peak_K"]-273.15, "overlap": rb["overlap"],
            "absorbed_power_W": rb["P_abs_W"],
        }])
        st.download_button("Download optimized point CSV", export.to_csv(index=False), "optimized_process_point.csv", "text/csv")

    st.markdown("### Pareto-style random screening")
    st.caption("Shows feasible candidates trading estimated build time against source energy. This is useful before committing to one weighted objective.")
    n_samples = st.slider("Random candidates", 200, 5000, 1200, 200)
    if st.button("Generate feasible Pareto sample"):
        rng = np.random.default_rng(11)
        X = np.column_stack([
            rng.uniform(P_lo, P_hi, n_samples),
            rng.uniform(v_lo, v_hi, n_samples),
            rng.uniform(h_lo, h_hi, n_samples),
            rng.uniform(t_lo, t_hi, n_samples),
        ])
        recs: List[Dict[str, float]] = []
        for xx in X:
            _, rr, dd = evaluate_candidate(xx)
            if dd["p_depth"] == 0 and dd["p_overlap"] == 0 and dd["p_temp"] == 0 and dd["p_ne"] == 0:
                P, vv, hh, tt = xx
                time_s = dd["build_time_s"]
                source_energy_kWh = P * time_s / 3.6e6
                recs.append({
                    "power_W": P, "speed_mm_s": vv, "hatch_mm": hh, "layer_mm": tt,
                    "time_min": time_s/60.0, "source_energy_kWh": source_energy_kWh,
                    "depth_mm": rr["depth_m"]*1e3, "overlap_pct": rr["overlap"]*100,
                })
        if recs:
            sdf = pd.DataFrame(recs)
            pf = pareto_front(sdf, "time_min", "source_energy_kWh")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sdf["time_min"], y=sdf["source_energy_kWh"], mode="markers", name="Feasible samples", opacity=0.35))
            fig.add_trace(go.Scatter(x=pf["time_min"], y=pf["source_energy_kWh"], mode="lines+markers", name="Non-dominated front"))
            fig.update_layout(xaxis_title="Estimated build time (min)", yaxis_title="Source energy (kWh)", height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pf.head(30), use_container_width=True, hide_index=True)
        else:
            st.warning("No feasible points found in this random sample. Widen the bounds or relax the active constraints.")

with tab_powder:
    st.markdown("## Powder size, wavelength, and coupling diagnostics")
    d10 = st.number_input("Powder D10 (µm)", 0.01, 5000.0, 15.0, 1.0)
    d50 = st.number_input("Powder D50 (µm)", 0.01, 5000.0, 35.0, 1.0)
    d90 = st.number_input("Powder D90 (µm)", 0.01, 5000.0, 55.0, 1.0)
    sphericity = st.slider("Mean sphericity", 0.10, 1.00, 0.90, 0.01)
    cv_size = st.number_input("Size coefficient of variation (fraction)", 0.0, 2.0, 0.30, 0.05)

    if method == "Microwave":
        chi10 = (d10 * 1e-6 / 2.0) / max(result["skin_depth_m"], 1e-30)
        chi50 = (d50 * 1e-6 / 2.0) / max(result["skin_depth_m"], 1e-30)
        chi90 = (d90 * 1e-6 / 2.0) / max(result["skin_depth_m"], 1e-30)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("D50 radius / skin depth", f"{chi50:.2f}")
        p2.metric("λ0", f"{result['wavelength_m']*1e3:.2f} mm")
        p3.metric("rspot / λ0", f"{result['beam_radius_m']/result['wavelength_m']:.4f}")
        p4.metric("Far-field Airy radius (NA≈1)", f"{result['airy_radius_m']*1e3:.1f} mm")
        st.write(
            "The particle-radius / bulk-skin-depth ratio is a useful screening number, but it does not by itself determine bed absorption. "
            "Loose powder behaves differently from bulk metal because electrical contacts, oxide films, voids, multiple scattering, magnetic response, "
            "temperature, and possible micro-discharges alter coupling."
        )
        if result["beam_radius_m"] < 0.1 * result["wavelength_m"]:
            st.warning(
                "The selected microwave spot is far below the free-space wavelength. Treat this as a near-field, resonant, waveguide/aperture, or cavity-focusing requirement rather than ordinary far-field focusing."
            )

        freq = np.logspace(math.log10(0.3), math.log10(300.0), 250) * 1e9
        skin_um = np.array([skin_depth(ff, mat.mu_r, mat.sigma_e) for ff in freq]) * 1e6
        wave_mm = C0 / freq * 1e3
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq/1e9, y=skin_um, name="Bulk skin depth (µm)"))
        fig.update_xaxes(type="log", title="Frequency (GHz)")
        fig.update_yaxes(type="log", title="Skin depth (µm)")
        fig.update_layout(height=430, title="Bulk-metal skin depth vs frequency")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freq/1e9, y=wave_mm, name="Free-space wavelength"))
        fig2.update_xaxes(type="log", title="Frequency (GHz)")
        fig2.update_yaxes(type="log", title="λ0 (mm)")
        fig2.update_layout(height=430, title="Free-space wavelength vs frequency")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write(
            "For laser PBF, D10/D50/D90, sphericity, oxide condition, packing density and recycled-powder history affect spreadability, local absorptivity, "
            "effective conductivity, pore formation and track consistency. This app represents those effects only through editable packing fraction, absorptivity, "
            "thermal properties and the lumped energy-to-volume efficiency."
        )

    st.markdown("### Optional sintering-kinetics thermal dose")
    kc1, kc2 = st.columns(2)
    with kc1:
        Q_act_kJ_mol = st.number_input("Activation energy Qa (kJ/mol)", 1.0, 1000.0, 200.0, 10.0)
    with kc2:
        A0_rate = st.number_input("Pre-exponential rate A0 (1/s)", 1e-12, 1e20, 1e8, format="%.2e")
    k_sinter = A0_rate * math.exp(-(Q_act_kJ_mol * 1000.0) / max(R_GAS * result["T_target_K"], 1e-12))
    dose = k_sinter * result["dwell_s"]
    first_order_proxy = 1.0 - math.exp(-min(dose, 700.0))
    kc3, kc4 = st.columns(2)
    kc3.metric("Arrhenius rate proxy ks", f"{k_sinter:.3e} 1/s")
    kc4.metric("Single-pass kinetic conversion proxy", f"{100*first_order_proxy:.3f}%")
    st.caption("This is a thermal-dose comparison tool only. Replace A0, Qa and the conversion law with material- and mechanism-specific neck-growth/densification kinetics when data are available.")

    # Simple inconsistency index (diagnostic, not a constitutive law)
    inconsistency = cv_size * (1.0 + 2.0 * max(0.0, 0.9 - sphericity)) / max(packing_fraction, 0.05)
    st.metric("Powder inconsistency screening index", f"{inconsistency:.3f}")
    st.caption("Higher means greater expected sensitivity. This index is deliberately heuristic and should be calibrated to spreadability / density / melt-pool data.")

with tab_refs:
    st.markdown("## References used to structure the model")
    refs = [
        ("Rubenchik, King & Wu (2018), Scaling laws for additive manufacturing", "https://www.osti.gov/pages/biblio/1891730"),
        ("On the Fidelity of the Scaling Laws for Melt Pool Depth Analysis During LPBF (2022)", "https://doi.org/10.1007/s40192-022-00289-w"),
        ("Khairallah et al. (2016), LPBF melt-flow / pore / spatter physics", "https://arxiv.org/abs/1512.02593"),
        ("Rosenthal moving-source solution / heat-flow formulation", "https://link.springer.com/article/10.1007/s40194-018-0667-6"),
        ("ORNL (2024), extended Goldak heat-source model for LPBF", "https://www.ornl.gov/publication/gaussian-process-based-extended-goldak-heat-source-model-finite-element-simulation"),
        ("Review on Microwave–Matter Interaction Fundamentals (2016)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5502878/"),
        ("State-of-the-art in microwave processing of metals, powders and alloys (2024)", "https://www.sciencedirect.com/science/article/pii/S1364032124003769"),
        ("Microwave heating for sustainable material synthesis and processing (2026)", "https://www.mdpi.com/2076-3417/16/11/5198"),
        ("Microwave volumetric scaffolding + laser PBF of copper (2026)", "https://doi.org/10.1007/s40964-026-01738-0"),
        ("Absorptivity-calibrated normalized enthalpy for IN625 LPBF (2026)", "https://doi.org/10.1016/j.jmrt.2026.03.257"),
    ]
    for title, url in refs:
        st.markdown(f"- [{title}]({url})")

    st.markdown("## Important modeling limits")
    st.markdown(
        "- **Laser:** no recoil pressure, Marangoni convection, free-surface flow, vapor plume, keyhole multiple reflections, powder entrainment, spatter, or layer-history CFD.\n"
        "- **Microwave:** no full-wave Maxwell cavity solution, impedance matching network, resonant detuning, tooling/susceptor coupling, plasma/arcing, or temperature-dependent complex material properties.\n"
        "- **Powder:** no DEM spreading model, particle-scale contacts, oxidation kinetics, neck-growth/densification law, or porosity evolution.\n"
        "- **Mechanics:** thermal strain and stress are screening proxies, not residual-stress or distortion predictions.\n"
        "- **Optimization:** optimums are only as valid as the selected property data and calibrated efficiencies. Treat them as candidates for simulation and coupons."
    )

    st.markdown("## Recommended next fidelity levels")
    st.markdown(
        "1. Calibrate absorptivity / coupling and ηm from single-track or single-spot experiments.\n"
        "2. Replace constant material properties with T-dependent functions.\n"
        "3. Laser: add Rosenthal/Goldak or FE thermal field, then free-surface CFD for keyhole regimes.\n"
        "4. Microwave: couple Maxwell + heat transfer + sintering densification in COMSOL/CST/HFSS-class modeling.\n"
        "5. Fit the paraboloid L/W/d response surfaces to CT/metallography or in-situ melt-pool data.\n"
        "6. Move from penalty optimization to explicit nonlinear constraints / Bayesian optimization / surrogate-assisted Pareto search once experimental data exist."
    )

st.divider()
st.caption(
    "Engineering screening model only. High-power lasers and microwave systems present serious optical, RF, electrical, thermal, fire, pressure, and metal-powder hazards; "
    "machine design and experimentation require appropriate shielding, interlocks, ventilation/inerting, grounding, RF leakage control, and qualified safety review."
)
