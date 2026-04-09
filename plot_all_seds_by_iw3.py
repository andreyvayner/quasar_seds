#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from astropy.io import fits

EMISSION_LINES_UM = [
    ("Ly$\\alpha$", 0.1216),
    ("C IV", 0.1549),
    ("C III]", 0.1909),
    ("Mg II", 0.2798),
    ("[O II]", 0.3727),
    ("H$\\beta$", 0.4861),
    ("[O III]", 0.5007),
    ("H$\\alpha$", 0.6563),
]


def gaussian_smooth_1d(y, sigma_pix):
    """Gaussian smooth a 1D array in pixel space using numpy convolution."""
    if sigma_pix is None or sigma_pix <= 0:
        return y

    radius = max(1, int(np.ceil(4.0 * sigma_pix)))
    xk = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xk / float(sigma_pix)) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def infer_error_column(names, y_label):
    """Infer error column name for the selected y column."""
    preferred = {
        "LUMINOSITY_NORM_W3": "LUMINOSITY_NORM_ERR_W3",
        "LUMINOSITY_ERG_S": "LUMINOSITY_ERR_ERG_S",
    }
    if y_label in preferred and preferred[y_label] in names:
        return preferred[y_label]

    generic = f"{y_label}_ERR"
    if generic in names:
        return generic

    return None


def adaptive_bin_to_snr(x, y, yerr, target_snr=5.0, min_bin_points=1):
    """Adaptive sequential binning so each bin reaches target SNR when possible."""
    if target_snr <= 0:
        return x, y, yerr
    if min_bin_points < 1:
        min_bin_points = 1

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    n = x.size
    if yerr.size != n:
        return x, y, yerr

    out_x = []
    out_y = []
    out_yerr = []

    i = 0
    while i < n:
        j = i
        sum_y = 0.0
        sum_var = 0.0
        has_err = False

        while j < n:
            yj = y[j]
            ej = yerr[j]
            if np.isfinite(yj):
                sum_y += yj
            if np.isfinite(ej) and ej > 0:
                sum_var += ej * ej
                has_err = True

            snr = (sum_y / np.sqrt(sum_var)) if (has_err and sum_var > 0) else np.nan
            j += 1
            nbin = j - i
            if has_err and np.isfinite(snr) and snr >= target_snr and nbin >= min_bin_points:
                break

        sl = slice(i, j)
        xb = x[sl]
        yb = y[sl]
        eb = yerr[sl]
        good = np.isfinite(xb) & np.isfinite(yb)
        if not np.any(good):
            i = j
            continue

        xb = xb[good]
        yb = yb[good]
        eb = eb[good]
        good_err = np.isfinite(eb) & (eb > 0)

        if np.any(good_err):
            w = 1.0 / (eb[good_err] ** 2)
            x_bin = np.sum(xb[good_err] * w) / np.sum(w)
            y_bin = np.average(yb[good_err], weights=w)
            yerr_bin = np.sqrt(np.sum(eb[good_err] ** 2)) / np.sum(good_err)
        else:
            x_bin = np.mean(xb)
            y_bin = np.mean(yb)
            yerr_bin = np.nan

        out_x.append(x_bin)
        out_y.append(y_bin)
        out_yerr.append(yerr_bin)
        i = j

    return np.array(out_x, dtype=float), np.array(out_y, dtype=float), np.array(out_yerr, dtype=float)


def outlier_mask_log_mad(x, y, window=11, sigma_thresh=5.0):
    """Return keep-mask from local MAD outlier rejection in log(y), evaluated in x-order."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 5 or window < 3 or sigma_thresh <= 0:
        return np.ones(n, dtype=bool)

    if window % 2 == 0:
        window += 1
    half = window // 2

    order = np.argsort(x)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    y_sorted = y[order]
    logy_sorted = np.log10(y_sorted)
    keep_sorted = np.ones(n, dtype=bool)
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        local = logy_sorted[i0:i1]
        if local.size <= 1:
            continue
        # Exclude the test point so a spike cannot dilute its own local statistics.
        local_wo_i = np.delete(local, i - i0)
        if local_wo_i.size < 3:
            continue
        med = np.nanmedian(local_wo_i)
        mad = np.nanmedian(np.abs(local_wo_i - med))
        if not np.isfinite(mad) or mad <= 0:
            local_std = np.nanstd(local_wo_i - med)
            if not np.isfinite(local_std) or local_std <= 0:
                continue
            robust_sigma = local_std
        else:
            robust_sigma = 1.4826 * mad
        if np.abs(logy_sorted[i] - med) > sigma_thresh * robust_sigma:
            keep_sorted[i] = False

    return keep_sorted[inv_order]


def get_iw3_ab(primary_header, table_header):
    """Return i-W3 (AB) color from primary or table header."""
    for hdr in (primary_header, table_header):
        if hdr is None:
            continue
        if "I_W3_AB" in hdr:
            val = hdr["I_W3_AB"]
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                return val
    return np.nan


def get_redshift(primary_header, table_header):
    """Return redshift from primary or table header."""
    for hdr in (primary_header, table_header):
        if hdr is None:
            continue
        if "REDSHIFT" in hdr:
            try:
                z = float(hdr["REDSHIFT"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(z) and z > -1.0:
                return z
    return np.nan


def load_sed(path, ycol, reject_outliers=True, outlier_window=11, outlier_sigma=5.0):
    """Load one SED FITS file and return wavelength, y, and color metadata."""
    with fits.open(path) as hdul:
        primary_header = hdul[0].header if len(hdul) > 0 else None
        fallback_header = hdul[1].header if len(hdul) > 1 else None
        color_val = get_iw3_ab(primary_header, fallback_header)
        z = get_redshift(primary_header, fallback_header)
        src_name = (
            primary_header.get("SRCNAME")
            or (fallback_header.get("SRCNAME") if fallback_header is not None else None)
            or os.path.basename(path).replace("_combined_sed.fits", "")
        )

        # Backward compatibility: old single-table format.
        if "COMBINED_SED" in hdul:
            table_hdu = hdul["COMBINED_SED"]
            data = table_hdu.data
            names = list(data.names)
            if "WAVELENGTH_UM" not in names:
                raise KeyError(f"{path}: missing WAVELENGTH_UM column")

            if ycol in names:
                y_label = ycol
            elif "LUMINOSITY_NORM_W3" in names:
                y_label = "LUMINOSITY_NORM_W3"
            elif "LUMINOSITY_ERG_S" in names:
                y_label = "LUMINOSITY_ERG_S"
            else:
                raise KeyError(f"{path}: no usable luminosity column found")

            x_obs = np.array(data["WAVELENGTH_UM"], dtype=float)
            origin = np.full(x_obs.size, "COMBINED", dtype="U16")
            y = np.array(data[y_label], dtype=float)
            yerr_col = infer_error_column(names, y_label)
            if yerr_col is not None:
                yerr = np.array(data[yerr_col], dtype=float)
            else:
                yerr = np.full_like(y, np.nan, dtype=float)
        else:
            # New format: combine dedicated HDUs.
            hdu_names = ["SDSS_SPEC", "SPHEREX_SPEC", "WISE_PHOT"]
            available = [name for name in hdu_names if name in hdul]
            if not available:
                raise KeyError(f"{path}: missing COMBINED_SED and dedicated HDUs")

            # Pick one y column to use consistently across all HDUs.
            y_candidates = [ycol]
            if "LUMINOSITY_NORM_W3" not in y_candidates:
                y_candidates.append("LUMINOSITY_NORM_W3")
            if "LUMINOSITY_ERG_S" not in y_candidates:
                y_candidates.append("LUMINOSITY_ERG_S")

            selected_y = None
            for cand in y_candidates:
                for hname in available:
                    data_h = hdul[hname].data
                    if data_h is None or len(data_h) == 0:
                        continue
                    names_h = list(data_h.names)
                    if cand in names_h and "WAVELENGTH_UM" in names_h:
                        selected_y = cand
                        break
                if selected_y is not None:
                    break

            if selected_y is None:
                raise KeyError(f"{path}: no usable luminosity column found in dedicated HDUs")

            y_label = selected_y
            x_chunks = []
            y_chunks = []
            yerr_chunks = []
            origin_chunks = []
            for hname in available:
                data_h = hdul[hname].data
                if data_h is None or len(data_h) == 0:
                    continue
                names_h = list(data_h.names)
                if "WAVELENGTH_UM" not in names_h or y_label not in names_h:
                    continue
                xh = np.array(data_h["WAVELENGTH_UM"], dtype=float)
                x_chunks.append(xh)
                y_chunks.append(np.array(data_h[y_label], dtype=float))
                yerr_col_h = infer_error_column(names_h, y_label)
                if yerr_col_h is not None:
                    yerr_chunks.append(np.array(data_h[yerr_col_h], dtype=float))
                else:
                    yerr_chunks.append(np.full(len(data_h), np.nan, dtype=float))
                origin_chunks.append(np.full(xh.size, hname, dtype="U16"))

            if not x_chunks:
                raise KeyError(f"{path}: no rows found in dedicated HDUs for column {y_label}")

            x_obs = np.concatenate(x_chunks)
            y = np.concatenate(y_chunks)
            yerr = np.concatenate(yerr_chunks)
            origin = np.concatenate(origin_chunks)

    x = np.array(x_obs, copy=True)
    if np.isfinite(z):
        x = x / (1.0 + z)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[good]
    x_obs = x_obs[good]
    y = y[good]
    yerr = yerr[good]
    origin = origin[good]

    removed_outliers = 0
    if reject_outliers and x.size > 0:
        keep = np.ones(x.size, dtype=bool)
        for origin_name in np.unique(origin):
            if origin_name == "WISE_PHOT":
                continue
            sel = origin == origin_name
            if np.count_nonzero(sel) < 5:
                continue
            idx_sel = np.where(sel)[0]
            keep_sel = np.ones(idx_sel.size, dtype=bool)

            # Two-pass clipping catches strong residual spikes, especially near edges.
            for _ in range(2):
                idx_active = idx_sel[keep_sel]
                if idx_active.size < 5:
                    break
                keep_active = outlier_mask_log_mad(
                    x[idx_active],
                    y[idx_active],
                    window=outlier_window,
                    sigma_thresh=outlier_sigma,
                )
                keep_next = np.zeros_like(keep_sel)
                keep_next[np.where(keep_sel)[0]] = keep_active
                if np.array_equal(keep_next, keep_sel):
                    break
                keep_sel = keep_next

            keep[idx_sel] = keep_sel

        removed_outliers = int(np.sum(~keep))
        x = x[keep]
        x_obs = x_obs[keep]
        y = y[keep]
        yerr = yerr[keep]
        origin = origin[keep]

    return {
        "path": path,
        "src_name": str(src_name),
        "x": x,
        "x_obs": x_obs,
        "y": y,
        "yerr": yerr,
        "origin": origin,
        "i_w3_ab": color_val,
        "z": z,
        "y_label": y_label,
        "n_outliers_removed": removed_outliers,
    }


def process_sed_for_plot(sed, args):
    """Apply adaptive binning/smoothing pipeline and return processed arrays."""
    x_plot = sed["x"]
    y_plot = sed["y"]
    yerr_plot = sed["yerr"]
    origin_plot = sed["origin"]
    x_proc = x_plot
    y_proc = y_plot
    yerr_proc = yerr_plot
    origin_proc = origin_plot

    x_parts = []
    y_parts = []
    origin_parts = []
    if args.adaptive_bin and np.any(np.isin(origin_proc, ["SDSS_SPEC", "SPHEREX_SPEC"])):
        for group_name in ["SDSS_SPEC", "SPHEREX_SPEC"]:
            sel = origin_proc == group_name
            if not np.any(sel):
                continue
            xg = x_proc[sel]
            yg = y_proc[sel]
            eg = yerr_proc[sel]
            if np.any(np.isfinite(eg) & (eg > 0)):
                xg, yg, _ = adaptive_bin_to_snr(
                    xg,
                    yg,
                    eg,
                    target_snr=args.target_snr,
                    min_bin_points=args.min_bin_points,
                )
            else:
                print(f"Warning: {sed['src_name']} {group_name} has no valid error column; skipping adaptive binning.")
            yg = gaussian_smooth_1d(yg, args.smooth_sigma)
            x_parts.append(xg)
            y_parts.append(yg)
            origin_parts.append(np.full(xg.size, group_name, dtype="U16"))

        # Keep non-spectrum points unbinned in adaptive mode.
        other = ~np.isin(origin_proc, ["SDSS_SPEC", "SPHEREX_SPEC"])
        if np.any(other):
            xo = x_proc[other]
            yo = gaussian_smooth_1d(y_proc[other], args.smooth_sigma)
            x_parts.append(xo)
            y_parts.append(yo)
            origin_parts.append(origin_proc[other])
    else:
        if args.adaptive_bin:
            if np.any(np.isfinite(yerr_proc) & (yerr_proc > 0)):
                x_proc, y_proc, _ = adaptive_bin_to_snr(
                    x_proc,
                    y_proc,
                    yerr_proc,
                    target_snr=args.target_snr,
                    min_bin_points=args.min_bin_points,
                )
            else:
                print(f"Warning: {sed['src_name']} has no valid error column; skipping adaptive binning.")
        y_proc = gaussian_smooth_1d(y_proc, args.smooth_sigma)
        x_parts.append(x_proc)
        y_parts.append(y_proc)
        origin_parts.append(origin_proc)

    x_out = np.concatenate(x_parts) if x_parts else x_proc
    y_out = np.concatenate(y_parts) if y_parts else y_proc
    origin_out = np.concatenate(origin_parts) if origin_parts else origin_proc
    if x_out.size > 1:
        srt = np.argsort(x_out)
        x_out = x_out[srt]
        y_out = y_out[srt]
        origin_out = origin_out[srt]
    return x_out, y_out, origin_out


def main():
    parser = argparse.ArgumentParser(
        description="Overlay all SED FITS files and color-code each SED by i-W3 (AB)."
    )
    parser.add_argument(
        "--pattern",
        default="*_combined_sed.fits",
        help="Glob pattern for SED FITS files (default: *_combined_sed.fits)",
    )
    parser.add_argument(
        "--ycol",
        default="LUMINOSITY_NORM_W3",
        help="Y-axis column to plot (default: LUMINOSITY_NORM_W3)",
    )
    parser.add_argument(
        "--output",
        default="all_seds_iW3_colored.png",
        help="Output figure filename (default: all_seds_iW3_colored.png)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma in pixels (default: 0 = no smoothing)",
    )
    parser.add_argument(
        "--adaptive-bin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turn adaptive spectral binning on/off (default: off)",
    )
    parser.add_argument(
        "--target-snr",
        type=float,
        default=3.0,
        help="Target SNR per adaptive bin (default: 5)",
    )
    parser.add_argument(
        "--min-bin-points",
        type=int,
        default=1,
        help="Minimum number of points per adaptive bin (default: 1)",
    )
    parser.add_argument(
        "--reject-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turn robust outlier rejection on/off during SED loading (default: on)",
    )
    parser.add_argument(
        "--outlier-window",
        type=int,
        default=11,
        help="Window size for local outlier detection in points (default: 11)",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=5.0,
        help="Sigma threshold for outlier rejection in log-flux space (default: 5)",
    )
    parser.add_argument(
        "--single-sed-test",
        default=None,
        help="Path to a single FITS SED file for original-vs-binned diagnostic plot",
    )
    args = parser.parse_args()

    if args.single_sed_test:
        sed = load_sed(
            args.single_sed_test,
            args.ycol,
            reject_outliers=args.reject_outliers,
            outlier_window=args.outlier_window,
            outlier_sigma=args.outlier_sigma,
        )
        x_bin, y_bin, origin_bin = process_sed_for_plot(sed, args)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(
            sed["x"],
            sed["y"],
            color="0.65",
            linewidth=1.0,
            alpha=0.5,
            label="Original",
        )
        ax.scatter(sed["x"], sed["y"], color="0.65", s=10, alpha=0.5)

        wise_mask = origin_bin == "WISE_PHOT"
        non_wise_mask = ~wise_mask
        if np.any(non_wise_mask):
            ax.plot(
                x_bin[non_wise_mask],
                y_bin[non_wise_mask],
                color="tab:blue",
                linewidth=1.4,
                alpha=0.5,
                label="Binned",
            )
        if np.any(wise_mask):
            ax.plot(
                x_bin[wise_mask],
                y_bin[wise_mask],
                linestyle="-",
                linewidth=1.2,
                color="tab:blue",
                marker="o",
                markersize=5.0,
                markerfacecolor="tab:blue",
                markeredgecolor="tab:blue",
                alpha=0.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rest-frame Wavelength (micron)", fontsize=14)
        ax.set_ylabel(sed["y_label"], fontsize=14)
        ax.set_title(f"{sed['src_name']}: Original vs Binned SED", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.legend()

        out = args.output
        if out == "all_seds_iW3_colored.png":
            stem = os.path.splitext(os.path.basename(args.single_sed_test))[0]
            out = f"{stem}_original_vs_binned.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved single-SED diagnostic plot: {out}")
        return

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    seds = []
    for path in paths:
        try:
            sed = load_sed(
                path,
                args.ycol,
                reject_outliers=args.reject_outliers,
                outlier_window=args.outlier_window,
                outlier_sigma=args.outlier_sigma,
            )
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if not np.isfinite(sed["z"]):
            print(f"Warning: {path} missing REDSHIFT; using observed-frame wavelengths.")
        if sed["x"].size == 0:
            print(f"Skipping {path}: no finite positive points in selected columns")
            continue
        if sed["n_outliers_removed"] > 0:
            print(f"{sed['src_name']}: removed {sed['n_outliers_removed']} outlier points")
        seds.append(sed)

    if not seds:
        raise RuntimeError("No usable SEDs found after reading files.")

    color_values = np.array([s["i_w3_ab"] for s in seds], dtype=float)
    finite_colors = color_values[np.isfinite(color_values)]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = colors.LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

    if finite_colors.size > 0:
        vmin = np.nanmin(finite_colors)
        vmax = np.nanmax(finite_colors)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
    else:
        norm = None
        sm = None

    for sed in seds:
        cval = sed["i_w3_ab"]
        if norm is not None and np.isfinite(cval):
            line_color = cmap(norm(cval))
        else:
            line_color = "0.75"
        alpha = 0.5

        x_plot, y_plot, origin_final = process_sed_for_plot(sed, args)

        wise_mask = origin_final == "WISE_PHOT"
        non_wise_mask = ~wise_mask
        if np.any(non_wise_mask):
            ax.plot(x_plot[non_wise_mask], y_plot[non_wise_mask], color=line_color, linewidth=1.3, alpha=alpha)
        if np.any(wise_mask):
            ax.plot(
                x_plot[wise_mask],
                y_plot[wise_mask],
                linestyle="-",
                linewidth=1.1,
                color=line_color,
                marker="o",
                markersize=5.5,
                markerfacecolor=line_color,
                markeredgecolor=line_color,
                alpha=alpha,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rest-frame Wavelength (micron)", fontsize=16)
    ax.set_ylabel(seds[0]["y_label"], fontsize=16)
    ax.set_title("Combined SEDs (rest-frame) colored by i-W3 (AB)", fontsize=17)
    ax.tick_params(axis="both", which="both", labelsize=13)
    ax.grid(True, alpha=0.3)

    # Label key emission lines (rest-frame wavelengths in micron).
    xmin, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    label_y = ymax / 1.6
    for line_name, line_um in EMISSION_LINES_UM:
        if xmin <= line_um <= xmax:
            ax.axvline(line_um, color="0.5", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0)
            ax.text(
                line_um * 1.01,
                label_y,
                line_name,
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=10,
                color="0.35",
            )

    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("i - W3 (AB mag)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {args.output}")
    print(f"Plotted {len(seds)} SEDs from pattern: {args.pattern}")


if __name__ == "__main__":
    main()
    #python3 plot_all_seds_by_iw3.py --single-sed-test J2346+1247_combined_sed.fits --output test_single_sed.png
