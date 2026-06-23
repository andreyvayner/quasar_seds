#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii, fits
from scipy.optimize import least_squares


C_KMS = 299792.458
CALZETTI_RV = 4.05
CASE_B_HYDROGEN_RATIOS = {
    "Hg_to_Hb": 1.0 / 2.13,
    "Hb_to_Hb": 1.0,
    "Ha_to_Hb": 2.86,
    "PaG_to_Hb": 1.0 / 31.8,
    "PaB_to_Hb": 1.81 / 31.8,
}

WAVE_COL_CANDIDATES = ("WAVELENGTH_UM", "WAVELENGTH", "lambda")
FLUX_COL_CANDIDATES = ("STACKED_FLUX", "FLUX_UJY", "FLUX", "flux")
ERR_COL_CANDIDATES = ("STACKED_ERR", "FLUX_ERR_UJY", "FLUX_ERR", "flux_err")
NCONTRIB_COL_CANDIDATES = ("N_CONTRIB",)


@dataclass(frozen=True)
class NarrowLine:
    name: str
    center_um: float
    amplitude_key: str
    group: str = "narrow"
    ratio_to: str | None = None
    ratio_value: float | None = None
    ratio_param: str | None = None


@dataclass(frozen=True)
class BroadHydrogenLine:
    name: str
    center_um: float
    amplitude_key: str
    ratio_to: str | None = None
    ratio_value: float | None = None
    ratio_param: str | None = None


    # NarrowLine("Ly_alpha_narrow", 0.121567, "amp_lya_n"),
    # NarrowLine("NV_1240", 0.1240, "amp_nv1240"),
    # NarrowLine("SiIV_OIV_1400", 0.1400, "amp_siiv_oiv"),
    # NarrowLine("CIV_1549", 0.1549, "amp_civ"),
    # NarrowLine("HeII_1640", 0.1640, "amp_heii1640"),
    # NarrowLine("CIII_1909", 0.1909, "amp_ciii"),
    #NarrowLine("PaB_narrow", 1.2818, "amp_pab_n")
NARROW_LINES = [
    # NarrowLine("MgII_2798", 0.2798, "amp_mgii"),
    NarrowLine("OII_3727", 0.3727, "amp_oii3727"),
    NarrowLine("Hg_narrow", 0.434047, "amp_hb_n", ratio_to="amp_hb_n", ratio_value=CASE_B_HYDROGEN_RATIOS["Hg_to_Hb"]),
    NarrowLine("Hb_narrow", 0.486133, "amp_hb_n"),
    NarrowLine("OIII_4959", 0.495891, "amp_oiii5007", ratio_to="amp_oiii5007", ratio_value=1.0 / 2.98),
    NarrowLine("OIII_5007", 0.500684, "amp_oiii5007"),
    NarrowLine("OI_6300", 0.630030, "amp_oi6300"),
    NarrowLine("NII_6548", 0.654805, "amp_nii6583", ratio_to="amp_nii6583", ratio_value=1.0 / 2.96),
    NarrowLine("Ha_narrow", 0.656281, "amp_hb_n", ratio_to="amp_hb_n", ratio_value=CASE_B_HYDROGEN_RATIOS["Ha_to_Hb"]),
    NarrowLine("NII_6583", 0.658345, "amp_nii6583"),
    NarrowLine("SII_6716", 0.671644, "amp_sii6716"),
    NarrowLine("SII_6731", 0.673082, "amp_sii6731"),
    # NarrowLine("HeI_10830", 1.0830, "amp_hei10830"),
    NarrowLine("PaG_narrow", 1.0938, "amp_hb_n", ratio_to="amp_hb_n", ratio_value=CASE_B_HYDROGEN_RATIOS["PaG_to_Hb"]),
    NarrowLine("PaB_narrow", 1.2818, "amp_hb_n", ratio_to="amp_hb_n", ratio_value=CASE_B_HYDROGEN_RATIOS["PaB_to_Hb"]),
]

HYDROGEN_BROAD_LINES = [
#    BroadHydrogenLine("Ly_alpha_broad", 0.121567, "amp_lya_b"),
    BroadHydrogenLine("Hg_broad", 0.434047, "amp_hb_b", ratio_to="amp_hb_b", ratio_value=CASE_B_HYDROGEN_RATIOS["Hg_to_Hb"]),
    BroadHydrogenLine("Hb_broad", 0.486133, "amp_hb_b"),
    BroadHydrogenLine("Ha_broad", 0.656281, "amp_hb_b", ratio_to="amp_hb_b", ratio_value=CASE_B_HYDROGEN_RATIOS["Ha_to_Hb"]),
    BroadHydrogenLine("HeI_10830_broad", 1.0830, "amp_hei10830_b"),
    BroadHydrogenLine("PaG_broad", 1.0938, "amp_hb_b", ratio_to="amp_hb_b", ratio_value=CASE_B_HYDROGEN_RATIOS["PaG_to_Hb"]),
    BroadHydrogenLine("PaB_broad", 1.2818, "amp_hb_b", ratio_to="amp_hb_b", ratio_value=CASE_B_HYDROGEN_RATIOS["PaB_to_Hb"]),
]

BASE_PARAMETER_SPECS = [
    ("cont_norm", 1.0, 1e-12, np.inf),
    ("cont_break_um", 0.45, 0.12, 2.5),
    ("cont_alpha_blue", -1.5, -6.0, 4.0),
    ("cont_alpha_red", -0.5, -6.0, 4.0),
    ("cont_smooth", 0.08, 0.005, 0.6),
    ("v_narrow_kms", 0.0, -3000.0, 3000.0),
    ("fwhm_narrow_kms", 1200.0, 50.0, 8000.0),
    ("v_broad_kms", 0.0, -3000.0, 3000.0),
    ("fwhm_broad_kms", 8000.0, 500.0, 25000.0),
]

AMP_PARAMETER_SPECS = [
    # ("amp_lya_n", 0.1, 0.0, np.inf),
    # ("amp_nv1240", 0.05, 0.0, np.inf),
    # ("amp_siiv_oiv", 0.05, 0.0, np.inf),
    # ("amp_civ", 0.08, 0.0, np.inf),
    # ("amp_heii1640", 0.03, 0.0, np.inf),
    # ("amp_ciii", 0.05, 0.0, np.inf),
    ("amp_mgii", 0.05, 0.0, np.inf),
    ("amp_oii3727", 0.03, 0.0, np.inf),
    ("amp_hb_n", 0.04, 0.0, np.inf),
    ("ebv_n", 0.1, 0.0, 1.0),
    ("amp_oiii5007", 0.08, 0.0, np.inf),
    ("amp_oi6300", 0.02, 0.0, np.inf),
    ("amp_nii6583", 0.04, 0.0, np.inf),
    ("amp_sii6716", 0.02, 0.0, np.inf),
    ("amp_sii6731", 0.02, 0.0, np.inf),
    # ("amp_hei10830", 0.03, 0.0, np.inf),
    # ("amp_lya_b", 0.15, 0.0, np.inf),
    ("amp_hb_b", 0.08, 0.0, np.inf),
    ("ebv_b", 0.2, 0.0, 1.0),
    ("amp_hei10830_b", 0.05, 0.0, np.inf),
]

ALL_PARAMETER_SPECS = BASE_PARAMETER_SPECS + AMP_PARAMETER_SPECS
IRON_TEMPLATE_PARAMETER_SPECS = [
    ("amp_iron_template", 0.05, 0.0, np.inf),
    ("v_iron_kms", 0.0, -5000.0, 5000.0),
    ("fwhm_iron_kms", 2500.0, 100.0, 15000.0),
]


def build_parameter_setup(include_iron_template: bool) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    parameter_specs = list(ALL_PARAMETER_SPECS)
    if include_iron_template:
        parameter_specs.extend(IRON_TEMPLATE_PARAMETER_SPECS)
    names = [item[0] for item in parameter_specs]
    p0 = np.array([item[1] for item in parameter_specs], dtype=float)
    lower = np.array([item[2] for item in parameter_specs], dtype=float)
    upper = np.array([item[3] for item in parameter_specs], dtype=float)
    param_index = {name: idx for idx, name in enumerate(names)}
    return names, p0, lower, upper, param_index


def _pick_column(colnames: list[str], candidates: tuple[str, ...]) -> str | None:
    name_set = set(colnames)
    for candidate in candidates:
        if candidate in name_set:
            return candidate
    return None


def read_stacked_spectrum(path: Path) -> dict[str, np.ndarray | float | bool]:
    with fits.open(path) as hdul:
        table_hdu = hdul["STACKED_SPHEREX"] if "STACKED_SPHEREX" in hdul else hdul[1]
        data = table_hdu.data
        colnames = list(data.names)

        wave_col = _pick_column(colnames, WAVE_COL_CANDIDATES)
        flux_col = _pick_column(colnames, FLUX_COL_CANDIDATES)
        err_col = _pick_column(colnames, ERR_COL_CANDIDATES)
        n_col = _pick_column(colnames, NCONTRIB_COL_CANDIDATES)

        if wave_col is None or flux_col is None or err_col is None:
            raise ValueError(f"Missing required columns in {path}: found {colnames}")

        wave = np.asarray(data[wave_col], dtype=float)
        flux = np.asarray(data[flux_col], dtype=float)
        err = np.asarray(data[err_col], dtype=float)
        n_contrib = np.asarray(data[n_col], dtype=float) if n_col is not None else np.full_like(wave, np.nan)
        rest_frame = bool(hdul[0].header.get("RESTFRM", True))

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return {
        "wave_um": wave[good],
        "flux": flux[good],
        "err": err[good],
        "n_contrib": n_contrib[good],
        "rest_frame": rest_frame,
    }


def read_template_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tab = ascii.read(path)
    names = list(tab.colnames)
    if len(names) < 2:
        raise ValueError(f"Template file {path} must contain at least two columns")
    wave = np.asarray(tab[names[0]], dtype=float)
    flux = np.asarray(tab[names[1]], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if wave.size < 2:
        raise ValueError(f"Template file {path} has too few valid points")
    return wave, flux


def smooth_broken_power_law(
    wave_um: np.ndarray,
    norm: float,
    break_um: float,
    alpha_blue: float,
    alpha_red: float,
    smooth: float,
) -> np.ndarray:
    x = np.asarray(wave_um, dtype=float) / break_um
    s = max(float(smooth), 1e-4)
    return norm * np.power(x, alpha_blue) * np.power(1.0 + np.power(x, 1.0 / s), (alpha_red - alpha_blue) * s)


def gaussian_profile(
    wave_um: np.ndarray,
    center_um: float,
    velocity_kms: float,
    fwhm_kms: float,
    amplitude: float,
) -> np.ndarray:
    shifted_center = center_um * (1.0 + velocity_kms / C_KMS)
    sigma_um = shifted_center * (fwhm_kms / 2.354820045) / C_KMS
    sigma_um = max(sigma_um, 1e-6)
    return amplitude * np.exp(-0.5 * ((wave_um - shifted_center) / sigma_um) ** 2)


def calzetti_k_lambda(wavelength_um: float) -> float:
    lam = float(wavelength_um)
    if 0.12 <= lam < 0.63:
        return 2.659 * (-2.156 + 1.509 / lam - 0.198 / lam**2 + 0.011 / lam**3) + CALZETTI_RV
    if 0.63 <= lam <= 2.2:
        return 2.659 * (-1.857 + 1.040 / lam) + CALZETTI_RV
    raise ValueError(f"Calzetti law not defined at wavelength {lam} um")


HBETA_REST_UM = 0.486133


def hydrogen_case_b_amplitude(
    anchor_amplitude: float,
    intrinsic_ratio: float,
    line_center_um: float,
    ebv: float,
) -> float:
    delta_k = calzetti_k_lambda(line_center_um) - calzetti_k_lambda(HBETA_REST_UM)
    attenuation_factor = np.power(10.0, -0.4 * ebv * delta_k)
    return anchor_amplitude * intrinsic_ratio * attenuation_factor


def build_iron_template_component(
    wave_um: np.ndarray,
    template_wave_um: np.ndarray,
    template_flux: np.ndarray,
    amplitude: float,
    velocity_kms: float,
    fwhm_kms: float,
) -> np.ndarray:
    shifted_wave = template_wave_um * (1.0 + velocity_kms / C_KMS)
    broaden_sigma = np.nanmedian(shifted_wave) * (fwhm_kms / 2.354820045) / C_KMS
    if not np.isfinite(broaden_sigma) or broaden_sigma <= 0:
        broaden_sigma = 0.0

    working_flux = np.array(template_flux, dtype=float)
    if broaden_sigma > 0:
        sigma_pix = broaden_sigma / np.nanmedian(np.diff(shifted_wave))
        if np.isfinite(sigma_pix) and sigma_pix > 0:
            half_width = max(3, int(np.ceil(4 * sigma_pix)))
            grid = np.arange(-half_width, half_width + 1, dtype=float)
            kernel = np.exp(-0.5 * (grid / sigma_pix) ** 2)
            kernel /= np.sum(kernel)
            working_flux = np.convolve(working_flux, kernel, mode="same")

    interp_flux = np.interp(wave_um, shifted_wave, working_flux, left=0.0, right=0.0)
    max_flux = np.nanmax(np.abs(interp_flux))
    if np.isfinite(max_flux) and max_flux > 0:
        interp_flux = interp_flux / max_flux
    return amplitude * interp_flux


def build_model_components(
    wave_um: np.ndarray,
    params: np.ndarray,
    parameter_names: list[str],
    param_index: dict[str, int],
    iron_template: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    p = {name: float(params[idx]) for name, idx in param_index.items()}

    continuum = smooth_broken_power_law(
        wave_um,
        p["cont_norm"],
        p["cont_break_um"],
        p["cont_alpha_blue"],
        p["cont_alpha_red"],
        p["cont_smooth"],
    )

    narrow_total = np.zeros_like(wave_um, dtype=float)
    broad_total = np.zeros_like(wave_um, dtype=float)
    narrow_by_line: dict[str, np.ndarray] = {}
    broad_by_line: dict[str, np.ndarray] = {}

    for line in NARROW_LINES:
        if line.ratio_to is not None and line.ratio_value is not None:
            amplitude = hydrogen_case_b_amplitude(
                anchor_amplitude=p[line.ratio_to],
                intrinsic_ratio=line.ratio_value,
                line_center_um=line.center_um,
                ebv=p["ebv_n"],
            )
        else:
            amplitude = p[line.amplitude_key]
        profile = gaussian_profile(wave_um, line.center_um, p["v_narrow_kms"], p["fwhm_narrow_kms"], amplitude)
        narrow_by_line[line.name] = profile
        narrow_total += profile

    for line in HYDROGEN_BROAD_LINES:
        if line.ratio_to is not None and line.ratio_value is not None:
            amplitude = hydrogen_case_b_amplitude(
                anchor_amplitude=p[line.ratio_to],
                intrinsic_ratio=line.ratio_value,
                line_center_um=line.center_um,
                ebv=p["ebv_b"],
            )
        else:
            amplitude = p[line.amplitude_key]
        profile = gaussian_profile(wave_um, line.center_um, p["v_broad_kms"], p["fwhm_broad_kms"], amplitude)
        broad_by_line[line.name] = profile
        broad_total += profile

    iron_template_component = np.zeros_like(wave_um, dtype=float)
    if iron_template is not None and "amp_iron_template" in param_index:
        iron_template_component = build_iron_template_component(
            wave_um,
            iron_template[0],
            iron_template[1],
            p["amp_iron_template"],
            p["v_iron_kms"],
            p["fwhm_iron_kms"],
        )

    emission_total = narrow_total + broad_total + iron_template_component
    model_total = continuum + emission_total
    return {
        "total": model_total,
        "continuum": continuum,
        "narrow_total": narrow_total,
        "broad_total": broad_total,
        "iron_template": iron_template_component,
        "narrow_by_line": narrow_by_line,
        "broad_by_line": broad_by_line,
    }


def residual_vector(
    params: np.ndarray,
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    parameter_names: list[str],
    param_index: dict[str, int],
    iron_template: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    model = build_model_components(wave_um, params, parameter_names, param_index, iron_template=iron_template)["total"]
    return (flux - model) / err


def estimate_initial_continuum(wave_um: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    wave_pivot = np.nanmedian(wave_um)
    cont_norm = np.nanmedian(flux)
    if not np.isfinite(cont_norm) or cont_norm <= 0:
        cont_norm = np.nanpercentile(flux, 60)
    if not np.isfinite(cont_norm) or cont_norm <= 0:
        cont_norm = 1.0
    if not np.isfinite(wave_pivot):
        wave_pivot = 0.45
    return cont_norm, wave_pivot


def prepare_initial_parameters(wave_um: np.ndarray, flux: np.ndarray, default_p0: np.ndarray, default_lower: np.ndarray, default_upper: np.ndarray, param_index: dict[str, int]) -> np.ndarray:
    p0 = default_p0.copy()
    cont_norm, wave_pivot = estimate_initial_continuum(wave_um, flux)
    p0[param_index["cont_norm"]] = cont_norm
    p0[param_index["cont_break_um"]] = np.clip(wave_pivot, default_lower[param_index["cont_break_um"]], default_upper[param_index["cont_break_um"]])
    return p0


def fit_stacked_spectrum(
    input_fits: str | Path,
    output_dir: str | Path | None = None,
    iron_template_path: str | Path | None = None,
    max_nfev: int = 200000,
    run_mcmc: bool = False,
    mcmc_nwalkers: int = 48,
    mcmc_nsteps: int = 12000,
    mcmc_burnin: int = 2000,
    mcmc_thin: int = 10,
) -> dict[str, object]:
    input_fits = Path(input_fits)
    data = read_stacked_spectrum(input_fits)
    if not data["rest_frame"]:
        raise ValueError(f"{input_fits} is not marked as rest frame; fit the rest-frame stack for this model.")

    wave_um = np.asarray(data["wave_um"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    err = np.asarray(data["err"], dtype=float)
    n_contrib = np.asarray(data["n_contrib"], dtype=float)
    iron_template = None
    if iron_template_path is not None:
        iron_template = read_template_ascii(Path(iron_template_path))
    parameter_names, default_p0, default_lower, default_upper, param_index = build_parameter_setup(iron_template is not None)

    if output_dir is None:
        output_dir = input_fits.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p0 = prepare_initial_parameters(wave_um, flux, default_p0, default_lower, default_upper, param_index)
    fit = least_squares(
        residual_vector,
        p0,
        args=(wave_um, flux, err, parameter_names, param_index, iron_template),
        bounds=(default_lower, default_upper),
        max_nfev=max_nfev,
    )

    n_points = wave_um.size
    n_params = fit.x.size
    dof = max(n_points - n_params, 1)
    chi2 = float(np.sum(fit.fun ** 2))
    redchi2 = chi2 / dof

    cov = None
    perr = np.full(n_params, np.nan, dtype=float)
    if fit.jac.size > 0:
        try:
            jac = fit.jac
            cov = np.linalg.inv(jac.T @ jac) * redchi2
            perr = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            cov = None

    model = build_model_components(wave_um, fit.x, parameter_names, param_index, iron_template=iron_template)
    continuum_subtracted = flux - model["continuum"]

    stem = input_fits.stem.replace(".fits", "")
    save_parameter_table(output_dir / f"{stem}_double_powerlaw_gaussian_fit.ecsv", parameter_names, fit.x, perr, chi2, redchi2, dof)
    save_model_fits(
        output_dir / f"{stem}_double_powerlaw_gaussian_model.fits",
        wave_um,
        flux,
        err,
        n_contrib,
        model,
        parameter_names,
        fit.x,
        chi2,
        redchi2,
        dof,
        iron_template_path,
    )
    save_diagnostic_plot(output_dir / f"{stem}_double_powerlaw_gaussian_fit.png", wave_um, flux, err, model, continuum_subtracted)
    save_component_plot(output_dir / f"{stem}_double_powerlaw_gaussian_components.png", wave_um, flux, err, model)
    save_zoomed_complex_plot(output_dir / f"{stem}_double_powerlaw_gaussian_zoomed_complexes.png", wave_um, flux, err, model)
    save_json_summary(output_dir / f"{stem}_double_powerlaw_gaussian_fit.json", parameter_names, fit.x, perr, chi2, redchi2, dof, iron_template_path)

    mcmc_result = None
    if run_mcmc:
        mcmc_result = run_emcee_after_least_squares(
            output_dir=output_dir,
            stem=stem,
            best_fit_params=fit.x,
            parameter_errors=perr,
            parameter_names=parameter_names,
            default_lower=default_lower,
            default_upper=default_upper,
            wave_um=wave_um,
            flux=flux,
            err=err,
            param_index=param_index,
            iron_template=iron_template,
            nwalkers=mcmc_nwalkers,
            nsteps=mcmc_nsteps,
            burnin=mcmc_burnin,
            thin=mcmc_thin,
        )

    return {
        "fit_result": fit,
        "parameter_names": parameter_names,
        "parameter_errors": perr,
        "chi2": chi2,
        "redchi2": redchi2,
        "dof": dof,
        "wave_um": wave_um,
        "flux": flux,
        "err": err,
        "model": model,
        "mcmc_result": mcmc_result,
    }


def run_emcee_after_least_squares(
    output_dir: Path,
    stem: str,
    best_fit_params: np.ndarray,
    parameter_errors: np.ndarray,
    parameter_names: list[str],
    default_lower: np.ndarray,
    default_upper: np.ndarray,
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    param_index: dict[str, int],
    iron_template: tuple[np.ndarray, np.ndarray] | None,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    thin: int,
) -> dict[str, object]:
    try:
        import emcee
    except ImportError as exc:
        raise ImportError("MCMC requested but emcee is not installed in this Python environment.") from exc

    ndim = best_fit_params.size
    if nwalkers < 2 * ndim:
        raise ValueError(f"mcmc_nwalkers must be at least {2 * ndim} for ndim={ndim}")
    if burnin >= nsteps:
        raise ValueError("mcmc_burnin must be smaller than mcmc_nsteps")
    if thin < 1:
        raise ValueError("mcmc_thin must be >= 1")

    def log_prior(theta: np.ndarray) -> float:
        if np.any(theta < default_lower) or np.any(theta > default_upper):
            return -np.inf
        return 0.0

    def log_probability(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        resid = residual_vector(theta, wave_um, flux, err, parameter_names, param_index, iron_template)
        return lp - 0.5 * np.sum(resid**2)

    rng = np.random.default_rng(42)
    finite_lower = np.where(np.isfinite(default_lower), default_lower, -1.0)
    finite_upper = np.where(np.isfinite(default_upper), default_upper, 1.0)
    finite_center = np.where(np.isfinite(best_fit_params), best_fit_params, 0.0)

    both_finite = np.isfinite(default_lower) & np.isfinite(default_upper)
    upper_only = (~np.isfinite(default_lower)) & np.isfinite(default_upper)
    lower_only = np.isfinite(default_lower) & (~np.isfinite(default_upper))
    neither_finite = (~np.isfinite(default_lower)) & (~np.isfinite(default_upper))

    finite_span = np.empty_like(best_fit_params, dtype=float)
    finite_span[both_finite] = np.maximum(default_upper[both_finite] - default_lower[both_finite], 1.0)
    finite_span[upper_only] = np.maximum(np.abs(default_upper[upper_only]), 1.0)
    finite_span[lower_only] = np.maximum(np.abs(default_lower[lower_only]), 1.0)
    finite_span[neither_finite] = np.maximum(np.abs(finite_center[neither_finite]), 1.0)

    init_scales = np.where(
        np.isfinite(parameter_errors) & (parameter_errors > 0),
        0.3 * parameter_errors,
        0.01 * finite_span,
    )
    init_scales = np.where(np.isfinite(init_scales) & (init_scales > 0), init_scales, 1e-3)
    init_scales = np.clip(init_scales, 1e-6, None)
    eps = 1e-8 * np.where(finite_span > 0, finite_span, 1.0)

    p0_walkers = initialize_mcmc_walkers(
        rng=rng,
        center=best_fit_params,
        init_scales=init_scales,
        lower=default_lower,
        upper=default_upper,
        eps=eps,
        nwalkers=nwalkers,
        ndim=ndim,
    )

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(p0_walkers, nsteps, progress=True, skip_initial_state_check=True)

    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    qminus = q50 - q16
    qplus = q84 - q50
    accept_frac = float(np.mean(sampler.acceptance_fraction))

    np.savez(
        output_dir / f"{stem}_double_powerlaw_gaussian_mcmc_samples.npz",
        samples=samples,
        parameter_names=np.array(parameter_names, dtype="U"),
        q16=q16,
        q50=q50,
        q84=q84,
        acceptance_fraction=accept_frac,
    )

    rows = []
    for name, p16, p50, p84, dm, dp in zip(parameter_names, q16, q50, q84, qminus, qplus):
        rows.append(
            {
                "parameter": name,
                "p16": float(p16),
                "p50": float(p50),
                "p84": float(p84),
                "err_minus": float(dm),
                "err_plus": float(dp),
            }
        )
    ascii.write(
        rows,
        output_dir / f"{stem}_double_powerlaw_gaussian_mcmc_summary.ecsv",
        format="ecsv",
        overwrite=True,
    )

    try:
        import corner
    except ImportError:
        corner_plot_path = None
    else:
        fig = corner.corner(
            samples,
            labels=parameter_names,
            truths=best_fit_params,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
        )
        corner_plot_path = output_dir / f"{stem}_double_powerlaw_gaussian_mcmc_corner.png"
        fig.savefig(corner_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return {
        "samples": samples,
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "acceptance_fraction": accept_frac,
        "corner_plot_path": corner_plot_path,
    }


def initialize_mcmc_walkers(
    rng: np.random.Generator,
    center: np.ndarray,
    init_scales: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: np.ndarray,
    nwalkers: int,
    ndim: int,
) -> np.ndarray:
    p0_walkers = None
    trial_scales = np.array(init_scales, dtype=float)
    safe_center = np.array(center, dtype=float)
    midpoint = np.zeros_like(safe_center, dtype=float)
    both_finite = np.isfinite(lower) & np.isfinite(upper)
    upper_only = (~np.isfinite(lower)) & np.isfinite(upper)
    lower_only = np.isfinite(lower) & (~np.isfinite(upper))
    neither_finite = (~np.isfinite(lower)) & (~np.isfinite(upper))
    midpoint[both_finite] = 0.5 * (lower[both_finite] + upper[both_finite])
    midpoint[upper_only] = upper[upper_only] - np.maximum(1.0, np.abs(upper[upper_only]) * 0.1)
    midpoint[lower_only] = lower[lower_only] + np.maximum(1.0, np.abs(lower[lower_only]) * 0.1)
    midpoint[neither_finite] = 0.0
    safe_center = np.where(np.isfinite(safe_center), safe_center, midpoint)
    lower_clip = np.where(np.isfinite(lower), lower + eps, -np.inf)
    upper_clip = np.where(np.isfinite(upper), upper - eps, np.inf)
    safe_center = np.clip(safe_center, lower_clip, upper_clip)
    safe_center = np.where(np.isfinite(safe_center), safe_center, 0.0)

    for _ in range(8):
        walkers = np.tile(safe_center, (nwalkers, 1))
        walkers += rng.normal(0.0, trial_scales, size=(nwalkers, ndim))
        walkers = np.clip(walkers, lower_clip, upper_clip)
        finite_mask = np.isfinite(walkers)
        if not np.all(finite_mask):
            fallback = np.tile(safe_center, (nwalkers, 1))
            walkers = np.where(finite_mask, walkers, fallback)

        centered = walkers - np.mean(walkers, axis=0, keepdims=True)
        try:
            rank = np.linalg.matrix_rank(centered)
        except np.linalg.LinAlgError:
            trial_scales *= 2.0
            continue
        if rank >= min(ndim, nwalkers - 1):
            p0_walkers = walkers
            break

        trial_scales *= 2.0

    if p0_walkers is None:
        # Final fallback: use a broader randomized cloud across the allowed box.
        p0_walkers = np.tile(safe_center, (nwalkers, 1))
        p0_walkers += rng.normal(0.0, np.maximum(trial_scales, 1e-3), size=(nwalkers, ndim))
        p0_walkers = np.clip(p0_walkers, lower_clip, upper_clip)
        finite_mask = np.isfinite(p0_walkers)
        if not np.all(finite_mask):
            fallback = np.tile(safe_center, (nwalkers, 1))
            p0_walkers = np.where(finite_mask, p0_walkers, fallback)

    return p0_walkers


def save_parameter_table(
    output_path: Path,
    parameter_names: list[str],
    params: np.ndarray,
    perr: np.ndarray,
    chi2: float,
    redchi2: float,
    dof: int,
) -> None:
    rows = []
    for name, value, error in zip(parameter_names, params, perr):
        rows.append({"parameter": name, "value": float(value), "error": float(error) if np.isfinite(error) else np.nan})
    rows.extend([
        {"parameter": "chi2", "value": float(chi2), "error": np.nan},
        {"parameter": "redchi2", "value": float(redchi2), "error": np.nan},
        {"parameter": "dof", "value": float(dof), "error": np.nan},
    ])
    ascii.write(rows, output_path, format="ecsv", overwrite=True)


def save_json_summary(
    output_path: Path,
    parameter_names: list[str],
    params: np.ndarray,
    perr: np.ndarray,
    chi2: float,
    redchi2: float,
    dof: int,
    iron_template_path: str | Path | None,
) -> None:
    summary = {
        "parameters": {
            name: {"value": float(value), "error": None if not np.isfinite(error) else float(error)}
            for name, value, error in zip(parameter_names, params, perr)
        },
        "chi2": float(chi2),
        "reduced_chi2": float(redchi2),
        "dof": int(dof),
        "iron_template": None if iron_template_path is None else str(iron_template_path),
        "fixed_doublet_ratios": {
            "[OIII]5007/[OIII]4959": 2.98,
            "[NII]6583/[NII]6548": 2.96,
        },
        "hydrogen_ratio_parameterization": {
            "anchor_line_narrow": "Hbeta",
            "anchor_line_broad": "Hbeta",
            "reddening_law": "Calzetti",
            "free_parameters": [
                "amp_hb_n",
                "ebv_n",
                "amp_hb_b",
                "ebv_b",
            ],
        },
        "case_b_intrinsic_hydrogen_ratios": {
            "Hg/Hb": CASE_B_HYDROGEN_RATIOS["Hg_to_Hb"],
            "Ha/Hb": CASE_B_HYDROGEN_RATIOS["Ha_to_Hb"],
            "PaG/Hb": CASE_B_HYDROGEN_RATIOS["PaG_to_Hb"],
            "PaB/Hb": CASE_B_HYDROGEN_RATIOS["PaB_to_Hb"],
        },
    }
    output_path.write_text(json.dumps(summary, indent=2))


def save_model_fits(
    output_path: Path,
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    n_contrib: np.ndarray,
    model: dict[str, np.ndarray | dict[str, np.ndarray]],
    parameter_names: list[str],
    params: np.ndarray,
    chi2: float,
    redchi2: float,
    dof: int,
    iron_template_path: str | Path | None,
) -> None:
    primary = fits.PrimaryHDU()
    primary.header["MODEL"] = ("DPL+GAUSS", "Smooth double power law plus Gaussian lines")
    primary.header["CHI2"] = (float(chi2), "Chi-square of best fit")
    primary.header["RCHI2"] = (float(redchi2), "Reduced chi-square")
    primary.header["DOF"] = (int(dof), "Degrees of freedom")
    primary.header["OIIIRAT"] = (2.98, "[OIII]5007/[OIII]4959 ratio")
    primary.header["NIIRAT"] = (2.96, "[NII]6583/[NII]6548 ratio")
    primary.header["HBANCHOR"] = ("TRUE", "Hydrogen lines anchored to Hbeta")
    primary.header["REDDLAW"] = ("CALZETTI", "Hydrogen attenuation law")
    primary.header["HGRAT"] = (CASE_B_HYDROGEN_RATIOS["Hg_to_Hb"], "Intrinsic case-B Hgamma/Hbeta ratio")
    primary.header["HARAT"] = (CASE_B_HYDROGEN_RATIOS["Ha_to_Hb"], "Intrinsic case-B Halpha/Hbeta ratio")
    primary.header["PAGRAT"] = (CASE_B_HYDROGEN_RATIOS["PaG_to_Hb"], "Intrinsic case-B Pagamma/Hbeta ratio")
    primary.header["PABRAT"] = (CASE_B_HYDROGEN_RATIOS["PaB_to_Hb"], "Intrinsic case-B Pabeta/Hbeta ratio")
    if iron_template_path is not None:
        primary.header["FETMPL"] = (str(iron_template_path)[:68], "Iron template file")
    for idx, name in enumerate(parameter_names[:90], start=1):
        primary.header[f"PAR{idx:02d}"] = f"{name}={params[idx - 1]:.8g}"

    model_cols = [
        fits.Column(name="WAVELENGTH_UM", array=wave_um, format="D", unit="um"),
        fits.Column(name="FLUX", array=flux, format="D"),
        fits.Column(name="FLUX_ERR", array=err, format="D"),
        fits.Column(name="N_CONTRIB", array=n_contrib.astype(np.int32), format="J"),
        fits.Column(name="MODEL_TOTAL", array=np.asarray(model["total"], dtype=float), format="D"),
        fits.Column(name="CONTINUUM", array=np.asarray(model["continuum"], dtype=float), format="D"),
        fits.Column(name="NARROW_TOTAL", array=np.asarray(model["narrow_total"], dtype=float), format="D"),
        fits.Column(name="BROAD_TOTAL", array=np.asarray(model["broad_total"], dtype=float), format="D"),
        fits.Column(name="IRON_TEMPLATE", array=np.asarray(model["iron_template"], dtype=float), format="D"),
        fits.Column(name="LINE_ONLY", array=np.asarray(model["narrow_total"], dtype=float) + np.asarray(model["broad_total"], dtype=float), format="D"),
    ]
    hdus = [primary, fits.BinTableHDU.from_columns(model_cols, name="MODEL")]

    narrow_by_line = model["narrow_by_line"]
    broad_by_line = model["broad_by_line"]
    for component_name, component_flux in list(narrow_by_line.items()) + list(broad_by_line.items()):
        cols = [
            fits.Column(name="WAVELENGTH_UM", array=wave_um, format="D", unit="um"),
            fits.Column(name="FLUX", array=np.asarray(component_flux, dtype=float), format="D"),
        ]
        ext_name = component_name.upper()[:68]
        hdus.append(fits.BinTableHDU.from_columns(cols, name=ext_name))

    fits.HDUList(hdus).writeto(output_path, overwrite=True)


def save_diagnostic_plot(
    output_path: Path,
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    model: dict[str, np.ndarray | dict[str, np.ndarray]],
    continuum_subtracted: np.ndarray,
) -> None:
    total = np.asarray(model["total"], dtype=float)
    continuum = np.asarray(model["continuum"], dtype=float)
    iron_template = np.asarray(model["iron_template"], dtype=float)
    line_only = np.asarray(model["narrow_total"], dtype=float) + np.asarray(model["broad_total"], dtype=float) + iron_template

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.05},
    )

    ax0.plot(wave_um, flux, color="black", lw=1.3, label="Stacked spectrum")
    ax0.fill_between(wave_um, flux - err, flux + err, color="gray", alpha=0.18, linewidth=0)
    ax0.plot(wave_um, total, color="tab:red", lw=1.4, label="Total fit")
    ax0.plot(wave_um, continuum, color="tab:blue", lw=1.2, ls="--", label="Continuum")
    ax0.plot(wave_um, line_only + continuum, color="tab:orange", lw=1.0, alpha=0.8, label="Continuum + lines")
    if np.any(np.abs(iron_template) > 0):
        ax0.plot(wave_um, iron_template + continuum, color="tab:purple", lw=1.0, ls=":", label="Continuum + iron template")
    ax0.set_ylabel("Stacked Flux")
    ax0.legend(loc="best", fontsize=9)
    ax0.grid(alpha=0.25)

    ax1.plot(wave_um, continuum_subtracted, color="black", lw=1.1, label="Continuum-subtracted data")
    ax1.plot(wave_um, line_only, color="tab:red", lw=1.2, label="Emission-line model")
    ax1.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax1.set_xlabel("Rest Wavelength (um)")
    ax1.set_ylabel("Line Flux")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_component_plot(
    output_path: Path,
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    model: dict[str, np.ndarray | dict[str, np.ndarray]],
) -> None:
    continuum = np.asarray(model["continuum"], dtype=float)
    line_data = flux - continuum
    line_err = err
    total = np.asarray(model["total"], dtype=float)
    line_model_total = total - continuum
    iron_template = np.asarray(model["iron_template"], dtype=float)
    narrow_by_line = model["narrow_by_line"]
    broad_by_line = model["broad_by_line"]

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.06},
    )
    ax.plot(wave_um, line_data, color="black", lw=1.2, label="Continuum-subtracted stack")
    ax.fill_between(wave_um, line_data - line_err, line_data + line_err, color="gray", alpha=0.15, linewidth=0)
    ax.plot(wave_um, line_model_total, color="tab:red", lw=1.8, label="Total best-fit line model")
    if np.any(np.abs(iron_template) > 0):
        ax.plot(wave_um, iron_template, color="tab:purple", lw=1.4, ls=":", label="Iron template")

    color_map = plt.cm.get_cmap("tab20", len(narrow_by_line) + len(broad_by_line) + 2)

    for idx, (name, component) in enumerate(narrow_by_line.items()):
        component = np.asarray(component, dtype=float)
        if not np.any(np.isfinite(component)) or np.nanmax(component) <= 0:
            continue
        color = color_map(idx)
        ax.plot(wave_um, component, color=color, lw=1.1, alpha=0.95, label=name.replace("_", " "))
        peak_idx = int(np.nanargmax(component))
        ax.axvline(wave_um[peak_idx], color=color, lw=0.8, ls="--", alpha=0.5)

    offset = len(narrow_by_line)
    for idx, (name, component) in enumerate(broad_by_line.items()):
        component = np.asarray(component, dtype=float)
        if not np.any(np.isfinite(component)) or np.nanmax(component) <= 0:
            continue
        color = color_map(offset + idx)
        ax.plot(wave_um, component, color=color, lw=1.4, ls="--", alpha=0.95, label=name.replace("_", " "))
        peak_idx = int(np.nanargmax(component))
        ax.axvline(wave_um[peak_idx], color=color, lw=0.8, ls="--", alpha=0.5)

    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Line Flux")
    ax.set_title("Gaussian Emission-Line Components")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    residual = line_data - line_model_total
    ax_res.plot(wave_um, residual, color="black", lw=1.0)
    ax_res.fill_between(
        wave_um,
        residual - line_err,
        residual + line_err,
        color="gray",
        alpha=0.15,
        linewidth=0,
    )
    ax_res.axhline(0.0, color="tab:red", lw=1.0, ls="--")
    ax_res.set_xlabel("Rest Wavelength (um)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_zoomed_complex_plot(
    output_path: Path,
    wave_um: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    model: dict[str, np.ndarray | dict[str, np.ndarray]],
) -> None:
    continuum = np.asarray(model["continuum"], dtype=float)
    total = np.asarray(model["total"], dtype=float)
    line_data = flux - continuum
    line_err = err
    line_model_total = total - continuum
    iron_template = np.asarray(model["iron_template"], dtype=float)
    narrow_by_line = model["narrow_by_line"]
    broad_by_line = model["broad_by_line"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 7.5),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.06, "wspace": 0.25},
    )
    complexes = [
        ("Hγ + Hβ + [O III]", 0.425, 0.525, ["Hg_narrow", "Hg_broad", "Hb_narrow", "Hb_broad", "OIII_4959", "OIII_5007"]),
        ("Hα + [N II] + [S II] + [O I]", 0.615, 0.71, ["OI_6300", "NII_6548", "Ha_narrow", "Ha_broad", "NII_6583", "SII_6716", "SII_6731"]),
    ]

    color_map = plt.cm.get_cmap("tab20", len(narrow_by_line) + len(broad_by_line) + 2)
    component_colors: dict[str, tuple] = {}
    ordered_names = list(narrow_by_line.keys()) + list(broad_by_line.keys())
    for idx, name in enumerate(ordered_names):
        component_colors[name] = color_map(idx)

    for col, (title, xmin, xmax, focus_names) in enumerate(complexes):
        ax = axes[0, col]
        ax_res = axes[1, col]
        mask = (wave_um >= xmin) & (wave_um <= xmax)
        if not np.any(mask):
            continue

        ax.plot(wave_um[mask], line_data[mask], color="black", lw=1.2, label="Continuum-subtracted stack")
        ax.fill_between(
            wave_um[mask],
            line_data[mask] - line_err[mask],
            line_data[mask] + line_err[mask],
            color="gray",
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(wave_um[mask], line_model_total[mask], color="tab:red", lw=1.8, label="Total best-fit line model")
        if np.any(np.abs(iron_template[mask]) > 0):
            ax.plot(wave_um[mask], iron_template[mask], color="tab:purple", lw=1.2, ls=":", label="Iron template")

        for name in focus_names:
            if name in narrow_by_line:
                component = np.asarray(narrow_by_line[name], dtype=float)
                style = "-"
            elif name in broad_by_line:
                component = np.asarray(broad_by_line[name], dtype=float)
                style = "--"
            else:
                continue

            color = component_colors[name]
            ax.plot(wave_um[mask], component[mask], color=color, lw=1.3, ls=style, alpha=0.95, label=name.replace("_", " "))
            if np.any(np.isfinite(component[mask])) and np.nanmax(component[mask]) > 0:
                local_idx = np.where(mask)[0][int(np.nanargmax(component[mask]))]
                ax.axvline(wave_um[local_idx], color=color, lw=0.8, ls="--", alpha=0.5)

        residual = line_data[mask] - line_model_total[mask]
        ax_res.plot(wave_um[mask], residual, color="black", lw=1.0)
        ax_res.fill_between(
            wave_um[mask],
            residual - line_err[mask],
            residual + line_err[mask],
            color="gray",
            alpha=0.15,
            linewidth=0,
        )
        ax_res.axhline(0.0, color="tab:red", lw=1.0, ls="--")
        ax_res.set_xlim(xmin, xmax)
        ax_res.set_xlabel("Rest Wavelength (um)")
        ax_res.set_ylabel("Residual")
        ax_res.grid(alpha=0.25)

        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_xlim(xmin, xmax)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=False)

    axes[0, 0].set_ylabel("Line Flux")
    axes[0, 1].set_ylabel("Line Flux")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a stacked SPHEREx spectrum with a smooth double power-law continuum "
            "plus Gaussian emission-line components. Hydrogen lines receive broad BLR "
            "components, while a single narrow kinematic component is tied across the full line set."
        )
    )
    parser.add_argument("input_fits", help="Rest-frame stacked spectrum FITS from stack_spherex_spectra.py")
    parser.add_argument("--output-dir", default=None, help="Directory for fit products. Defaults next to the input FITS.")
    parser.add_argument("--iron-template", default=None, help="Optional ASCII Fe-emission template (first col wavelength [um], second col relative flux).")
    parser.add_argument("--max-nfev", type=int, default=200000, help="Maximum least-squares function evaluations.")
    parser.add_argument("--mcmc", action="store_true", help="Run an MCMC posterior exploration after the least-squares fit.")
    parser.add_argument("--mcmc-nwalkers", type=int, default=48, help="Number of emcee walkers.")
    parser.add_argument("--mcmc-nsteps", type=int, default=12000, help="Number of MCMC steps per walker.")
    parser.add_argument("--mcmc-burnin", type=int, default=2000, help="Burn-in steps discarded from each walker.")
    parser.add_argument("--mcmc-thin", type=int, default=10, help="Thinning factor applied when saving flattened samples.")
    args = parser.parse_args()

    result = fit_stacked_spectrum(
        args.input_fits,
        output_dir=args.output_dir,
        iron_template_path=args.iron_template,
        max_nfev=args.max_nfev,
        run_mcmc=args.mcmc,
        mcmc_nwalkers=args.mcmc_nwalkers,
        mcmc_nsteps=args.mcmc_nsteps,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thin=args.mcmc_thin,
    )
    print(f"Fit complete for {args.input_fits}")
    print(f"chi2 = {result['chi2']:.3f}")
    print(f"reduced chi2 = {result['redchi2']:.3f}")
    print(f"dof = {result['dof']}")
    if result["mcmc_result"] is not None:
        print(f"MCMC acceptance fraction = {result['mcmc_result']['acceptance_fraction']:.3f}")


if __name__ == "__main__":
    main()
