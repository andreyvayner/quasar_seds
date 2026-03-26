import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.cosmology import Planck18
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
from astroquery.ipac.irsa import Irsa
from astroquery.svo_fps import SvoFps
from astropy.io import ascii, fits

# ------------------------------------------------------------
# Inputs CHANGE HERE
# ------------------------------------------------------------
# src_name = 'J0006+1215'
# ipac_file = "J0006+1215.tbl"
# ra_str = "00:06:10.67"
# dec_str = "+12:15:01.2"
# z = 2.3184
# spec_source = 'sdss'
 
# src_name = 'J1451+233'
# ipac_file = "J1451+233.tbl"
# ra_str = "14:51:48.01"
# dec_str = "+23:38:45.4"
# z = 2.6348 


# src_name = 'J1232+0912'
# ipac_file = "J1232+0912.tbl"
# ra_str = "12:32:41.75  "
# dec_str = "09:12:09.3"
# z = 2.4034 


# src_name = 'J1145+5742'
# ipac_file = "J1145+5742.tbl"
# ra_str = "11:45:08.00  "
# dec_str = " 57:42:58.6"
# z = 2.8747 

 
# src_name = 'J0834+0159'
# ipac_file = "J0834+0159.tbl"
# ra_str = "08:34:48.48  "
# dec_str = "01:59:21.2 "
# z = 2.588


# src_name = 'J0220+0137'
# ipac_file ='J0220+0137.tbl'
# ra_str = '02:20:52.11'
# dec_str = '01:37:11.1'
# z = 3.1375 
# spec_source = "sdss"  # options: "sdss", "desi"

# src_name = 'J1705+2736'
# ipac_file = 'J1705+2736.tbl'
# ra_str ='17:05:58.56'
# dec_str = '27:36:24.7'
# z = 2.4461 
#spec_source = "sdss"  # options: "sdss", "desi"

# src_name = 'J1652+1728'
# ipac_file = 'J1652+1728.tbl'
# ra_str ='16:52:02.61 '
# dec_str = '17:28:52.3'
# z = 2.94
# spec_source = "sdss"  # options: "sdss", "desi"

# src_name = '3C298'
# ipac_file = '3C298.tbl'
# ra_str ='14 19 08.1802187928' #right ascension
# dec_str = '+06 28 34.802854680' #declination
# z = 1.439
# spec_source = "sdss"  # options: "sdss", "desi"

src_name = 'J0052+0101'
ipac_file = 'J0052+0101.tbl'
ra_str ='00:52:02.4056270064' #right ascension
dec_str = '+01:01:29.270570160' #declination
z = 2.27
spec_source = "desi"  # options: "sdss", "desi"






# ------------------------------------------------------------
# Read IPAC table
# Expected columns from your file:
#   lambda    [um]
#   flux      [uJy]
#   flux_err  [uJy]
# ------------------------------------------------------------
ipac_table = ascii.read(ipac_file, format="ipac")

wav_ipac = np.array(ipac_table["lambda"], dtype=float)     # micron
flux_ipac = np.array(ipac_table["flux"], dtype=float)      # microJy
err_ipac = np.array(ipac_table["flux_err"], dtype=float)   # microJy

# Sort for plotting
srt = np.argsort(wav_ipac)
wav_ipac = wav_ipac[srt]
flux_ipac = flux_ipac[srt]
err_ipac = err_ipac[srt]

# Source coordinate
coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")

# ------------------------------------------------------------
# Query SDSS photometry (u, g, r, i, z)
# ------------------------------------------------------------
sdss_photo = SDSS.query_region(
    coord,
    radius=5 * u.arcsec,
    photoobj_fields=[
        "ra", "dec",
        "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
        "psfMagErr_u", "psfMagErr_g", "psfMagErr_r", "psfMagErr_i", "psfMagErr_z",
    ],
    data_release=17
)

sdss_phot_bands = ["u", "g", "r", "i", "z"]
sdss_phot_wav_um = np.array([0.3551, 0.4686, 0.6165, 0.7481, 0.8931], dtype=float)
sdss_phot_flux_uJy = np.array([], dtype=float)
sdss_phot_err_uJy = np.array([], dtype=float)
sdss_phot_wav_used_um = np.array([], dtype=float)
sdss_phot_bands_used = []

if sdss_photo is None or len(sdss_photo) == 0:
    print("No SDSS photometric source found within 5 arcsec.")
else:
    photo_coords = SkyCoord(
        np.array(sdss_photo["ra"], dtype=float) * u.deg,
        np.array(sdss_photo["dec"], dtype=float) * u.deg,
        frame="icrs"
    )
    photo_sep = coord.separation(photo_coords)
    photo_best = sdss_photo[np.argmin(photo_sep)]

    fluxes = []
    errs = []
    used_wavs = []
    used_bands = []
    for b, w in zip(sdss_phot_bands, sdss_phot_wav_um):
        mcol = f"psfMag_{b}"
        ecol = f"psfMagErr_{b}"
        mag = photo_best[mcol]
        emag = photo_best[ecol]
        if np.ma.is_masked(mag) or not np.isfinite(mag):
            continue

        # SDSS magnitudes are on the AB system
        f_uJy = 10 ** ((23.9 - float(mag)) / 2.5)
        if np.ma.is_masked(emag) or not np.isfinite(emag):
            ef_uJy = np.nan
        else:
            ef_uJy = f_uJy * (np.log(10.0) / 2.5) * float(emag)

        fluxes.append(f_uJy)
        errs.append(ef_uJy)
        used_wavs.append(w)
        used_bands.append(b)

    sdss_phot_flux_uJy = np.array(fluxes, dtype=float)
    sdss_phot_err_uJy = np.array(errs, dtype=float)
    sdss_phot_wav_used_um = np.array(used_wavs, dtype=float)
    sdss_phot_bands_used = used_bands

# ------------------------------------------------------------
# Query IRSA AllWISE photometry
# ------------------------------------------------------------
wise_tab = Irsa.query_region(
    coordinates=coord,
    catalog="allwise_p3as_psd",
    spatial="Cone",
    radius=5 * u.arcsec
)

if wise_tab is None or len(wise_tab) == 0:
    print("No AllWISE source found within 5 arcsec.")
    wise_bands = []
    wise_wav_um = np.array([], dtype=float)
    wise_flux_uJy = np.array([], dtype=float)
    wise_flux_err_uJy = np.array([], dtype=float)
else:
    wise_coords = SkyCoord(
        np.array(wise_tab["ra"], dtype=float) * u.deg,
        np.array(wise_tab["dec"], dtype=float) * u.deg,
        frame="icrs"
    )
    wise_sep = coord.separation(wise_coords)
    wise_best = wise_tab[np.argmin(wise_sep)]

    # AllWISE Vega zero points in Jy and effective wavelengths in micron
    wise_meta = [
        ("W1", "w1mpro", "w1sigmpro", 3.4, 309.540),
        ("W2", "w2mpro", "w2sigmpro", 4.6, 171.787),
        ("W3", "w3mpro", "w3sigmpro", 12.0, 31.674),
        ("W4", "w4mpro", "w4sigmpro", 22.0, 8.363),
    ]

    wise_bands = []
    wise_wav_um = []
    wise_flux_uJy = []
    wise_flux_err_uJy = []

    for band, mag_col, emag_col, wave, f0_jy in wise_meta:
        mag = wise_best[mag_col]
        emag = wise_best[emag_col]
        if np.ma.is_masked(mag) or not np.isfinite(mag):
            continue

        flux_jy = f0_jy * 10 ** (-0.4 * float(mag))
        flux_ujy = flux_jy * 1e6

        if np.ma.is_masked(emag) or not np.isfinite(emag):
            err_ujy = np.nan
        else:
            # sigma_f/f = ln(10)/2.5 * sigma_m
            err_ujy = flux_ujy * (np.log(10.0) / 2.5) * float(emag)

        wise_bands.append(band)
        wise_wav_um.append(wave)
        wise_flux_uJy.append(flux_ujy)
        wise_flux_err_uJy.append(err_ujy)

    wise_wav_um = np.array(wise_wav_um, dtype=float)
    wise_flux_uJy = np.array(wise_flux_uJy, dtype=float)
    wise_flux_err_uJy = np.array(wise_flux_err_uJy, dtype=float)

# ------------------------------------------------------------
# Query and download optical spectrum (SDSS or DESI)
# ------------------------------------------------------------
spec_source = spec_source.lower().strip()

if spec_source == "sdss":
    matches = SDSS.query_region(
        coord,
        radius=5 * u.arcsec,
        spectro=True,
        data_release=17
    )

    if matches is None or len(matches) == 0:
        raise RuntimeError("No SDSS spectrum found within 5 arcsec.")

    match_coords = SkyCoord(
        np.array(matches["ra"], dtype=float) * u.deg,
        np.array(matches["dec"], dtype=float) * u.deg,
        frame="icrs"
    )
    sep = coord.separation(match_coords)
    best_idx = np.argmin(sep)
    best_match = matches[best_idx:best_idx + 1]

    print(f"Closest SDSS match separation: {sep[best_idx].arcsec:.3f} arcsec")

    spec_list = SDSS.get_spectra(matches=best_match, data_release=17)
    if spec_list is None or len(spec_list) == 0:
        raise RuntimeError("SDSS match found, but spectrum download failed.")

    hdul = spec_list[0]
    hdul.writeto(f"{src_name}_SDSS_spectrum.fits", overwrite=True)
    data = hdul[1].data

    loglam = np.array(data["loglam"], dtype=float)  # log10(lambda / Angstrom)
    flux_sdss = np.array(data["flux"], dtype=float)  # 1e-17 erg/s/cm^2/Angstrom
    ivar = np.array(data["ivar"], dtype=float)  # inverse variance
    wav_A = 10**loglam
    wav_um = wav_A * 1e-4
    spec_label = "SDSS spectrum"

elif spec_source == "desi":
    try:
        from sparcl.client import SparclClient
    except ImportError as exc:
        raise ImportError(
            "DESI mode requires sparclclient. Install with: pip install sparclclient"
        ) from exc

    client = SparclClient()
    search_radius_arcsec = 5.0
    radius_deg = search_radius_arcsec / 3600.0
    constraints = {
        "ra": [coord.ra.deg - radius_deg, coord.ra.deg + radius_deg],
        "dec": [coord.dec.deg - radius_deg, coord.dec.deg + radius_deg],
    }
    outfields = ["sparcl_id", "ra", "dec", "specprimary", "_dr"]
    found = client.find(outfields=outfields, constraints=constraints, limit=200)
    if found is None or len(found.records) == 0:
        raise RuntimeError("No DESI spectrum candidate found in SPARCL search window.")

    candidates = []
    for rec in found.records:
        if not (hasattr(rec, "ra") and hasattr(rec, "dec")):
            continue
        rcoord = SkyCoord(float(rec.ra) * u.deg, float(rec.dec) * u.deg, frame="icrs")
        rsep = coord.separation(rcoord).arcsec
        if np.isfinite(rsep) and rsep <= search_radius_arcsec:
            dataset_tag = str(getattr(rec, "_dr", ""))
            if "desi" not in dataset_tag.lower():
                continue
            is_primary = bool(getattr(rec, "specprimary", False))
            candidates.append((not is_primary, rsep, rec))

    if len(candidates) == 0:
        raise RuntimeError("DESI candidates found, but none are within 5 arcsec.")

    candidates.sort(key=lambda x: (x[0], x[1]))
    _, best_sep, best_rec = candidates[0]
    print(f"Closest DESI match separation: {best_sep:.3f} arcsec")

    include_fields = ["wavelength", "flux", "ivar", "specid", "targetid", "ra", "dec"]
    retrieve_kwargs = {"include": include_fields, "limit": 1}
    if hasattr(best_rec, "_dr") and best_rec._dr is not None:
        retrieve_kwargs["dataset_list"] = [best_rec._dr]
    retrieved = client.retrieve([best_rec.sparcl_id], **retrieve_kwargs)
    if retrieved is None or len(retrieved.records) == 0:
        raise RuntimeError("DESI match found, but spectrum retrieval failed.")

    record = retrieved.records[0]
    wav_A = np.array(record.wavelength, dtype=float)
    flux_sdss = np.array(record.flux, dtype=float)  # same f_lambda convention as SDSS
    if hasattr(record, "ivar") and record.ivar is not None:
        ivar = np.array(record.ivar, dtype=float)
    else:
        ivar = np.zeros_like(flux_sdss)

    cols = [
        fits.Column(name="WAVELENGTH", array=wav_A, format="D", unit="Angstrom"),
        fits.Column(name="FLUX", array=flux_sdss, format="D", unit="1e-17 erg s-1 cm-2 Angstrom-1"),
        fits.Column(name="IVAR", array=ivar, format="D"),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["SURVEY"] = "DESI"
    hdu.header["SOURCE"] = src_name
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f"{src_name}_DESI_spectrum.fits", overwrite=True)

    wav_um = wav_A * 1e-4
    spec_label = "DESI spectrum"
else:
    raise ValueError(f"Unsupported spec_source='{spec_source}'. Use 'sdss' or 'desi'.")

# ------------------------------------------------------------
# Convert optical f_lambda to microJy
#
# f_nu = f_lambda * lambda^2 / c
#
# lambda in Angstrom
# c = 2.99792458e18 Angstrom/s
# 1 microJy = 1e-29 erg/s/cm^2/Hz
# ------------------------------------------------------------
c_A_per_s = 2.99792458e18

f_lambda = flux_sdss * 1e-17
flux_sdss_uJy = (f_lambda * wav_A**2 / c_A_per_s) / 1e-29

# error from inverse variance
flux_sdss_err = np.full_like(flux_sdss, np.nan, dtype=float)
good_ivar = ivar > 0
flux_sdss_err[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar])

f_lambda_err = flux_sdss_err * 1e-17
flux_sdss_err_uJy = (f_lambda_err * wav_A**2 / c_A_per_s) / 1e-29

good_sdss = np.isfinite(wav_um) & np.isfinite(flux_sdss_uJy)

# ------------------------------------------------------------
# Convert f_nu (uJy) to nu*L_nu (erg/s)
# ------------------------------------------------------------
d_l = Planck18.luminosity_distance(z).to(u.cm)
nu_ipac = (c / (wav_ipac * u.um)).to(u.Hz)
nu_sdss = (c / (wav_um * u.um)).to(u.Hz)

fnu_ipac = flux_ipac * u.uJy
fnu_sdss = flux_sdss_uJy * u.uJy
fnu_wise = wise_flux_uJy * u.uJy
fnu_sdss_phot = sdss_phot_flux_uJy * u.uJy
err_fnu_ipac = err_ipac * u.uJy
err_fnu_sdss = flux_sdss_err_uJy * u.uJy
err_fnu_wise = wise_flux_err_uJy * u.uJy
err_fnu_sdss_phot = sdss_phot_err_uJy * u.uJy

# For observed flux density f_nu: nu*L_nu = 4*pi*d_L^2*(nu_obs*f_nu_obs)
nuLnu_ipac = (4 * np.pi * d_l**2 * nu_ipac * fnu_ipac).to(u.erg / u.s)
nuLnu_sdss = (4 * np.pi * d_l**2 * nu_sdss * fnu_sdss).to(u.erg / u.s)
nu_wise = (c / (wise_wav_um * u.um)).to(u.Hz)
nuLnu_wise = (4 * np.pi * d_l**2 * nu_wise * fnu_wise).to(u.erg / u.s)
nu_sdss_phot = (c / (sdss_phot_wav_used_um * u.um)).to(u.Hz)
nuLnu_sdss_phot = (4 * np.pi * d_l**2 * nu_sdss_phot * fnu_sdss_phot).to(u.erg / u.s)
err_nuLnu_ipac = (4 * np.pi * d_l**2 * nu_ipac * err_fnu_ipac).to(u.erg / u.s)
err_nuLnu_sdss = (4 * np.pi * d_l**2 * nu_sdss * err_fnu_sdss).to(u.erg / u.s)
err_nuLnu_wise = (4 * np.pi * d_l**2 * nu_wise * err_fnu_wise).to(u.erg / u.s)
err_nuLnu_sdss_phot = (4 * np.pi * d_l**2 * nu_sdss_phot * err_fnu_sdss_phot).to(u.erg / u.s)

# ------------------------------------------------------------
# Save combined SDSS/DESI + SPHEREx + all photometry points
# Columns:
#   wavelength_micron, luminosity_erg_s, luminosity_uncertainty_erg_s,
#   flux_uJy, flux_uncertainty_uJy
# ------------------------------------------------------------
wavelength_all = np.concatenate([
    wav_um[good_sdss],
    wav_ipac,
    wise_wav_um,
    sdss_phot_wav_used_um,
])
luminosity_all = np.concatenate([
    nuLnu_sdss.value[good_sdss],
    nuLnu_ipac.value,
    nuLnu_wise.value,
    nuLnu_sdss_phot.value,
])
luminosity_err_all = np.concatenate([
    err_nuLnu_sdss.value[good_sdss],
    err_nuLnu_ipac.value,
    err_nuLnu_wise.value,
    err_nuLnu_sdss_phot.value,
])
flux_all = np.concatenate([
    flux_sdss_uJy[good_sdss],
    flux_ipac,
    wise_flux_uJy,
    sdss_phot_flux_uJy,
])
flux_err_all = np.concatenate([
    flux_sdss_err_uJy[good_sdss],
    err_ipac,
    wise_flux_err_uJy,
    sdss_phot_err_uJy,
])

sort_idx = np.argsort(wavelength_all)
combined_out = np.column_stack([
    wavelength_all[sort_idx],
    luminosity_all[sort_idx],
    luminosity_err_all[sort_idx],
    flux_all[sort_idx],
    flux_err_all[sort_idx],
])

combined_csv = f"{src_name}_combined_sed.csv"
np.savetxt(
    combined_csv,
    combined_out,
    delimiter=",",
    header=(
        "wavelength_micron,luminosity_erg_s,luminosity_uncertainty_erg_s,"
        "flux_uJy,flux_uncertainty_uJy"
    ),
    comments="",
)
print(f"Saved combined SED points: {combined_csv}")

# ------------------------------------------------------------
# Plot 1: observed flux density (microJy) versus wavelength
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Optical spectrum
plt.plot(
    wav_um[good_sdss],
    flux_sdss_uJy[good_sdss],
    linewidth=1.0,
    label=spec_label
)

# IPAC points with error bars
plt.errorbar(
    wav_ipac,
    flux_ipac,
    yerr=err_ipac,
    fmt="o",
    markersize=4,
    capsize=3,
    linestyle="none",
    label="SPHEREx"
)

if wise_wav_um.size > 0:
    plt.errorbar(
        wise_wav_um,
        wise_flux_uJy,
        yerr=wise_flux_err_uJy,
        fmt="s",
        markersize=6,
        capsize=3,
        linestyle="none",
        color="tab:red",
        label="WISE (AllWISE)"
    )
    for i, band in enumerate(wise_bands):
        plt.text(wise_wav_um[i] * 1.03, wise_flux_uJy[i], band, color="tab:red", fontsize=9)

if sdss_phot_wav_used_um.size > 0:
    plt.errorbar(
        sdss_phot_wav_used_um,
        sdss_phot_flux_uJy,
        yerr=sdss_phot_err_uJy,
        fmt="^",
        markersize=6,
        capsize=3,
        linestyle="none",
        color="tab:green",
        label="SDSS photometry"
    )
    for i, band in enumerate(sdss_phot_bands_used):
        plt.text(sdss_phot_wav_used_um[i] * 1.03, sdss_phot_flux_uJy[i], band, color="tab:green", fontsize=9)

plt.xlabel("Wavelength (micron)")
plt.ylabel("Flux (microJy)")
plt.xscale('log')
plt.title(f"{src_name}: {spec_source.upper()} + IPAC Spectrum")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ------------------------------------------------------------
# Plot 2: nu*L_nu versus wavelength
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Optical spectrum
plt.plot(
    wav_um[good_sdss],
    nuLnu_sdss.value[good_sdss],
    linewidth=1.0,
    label=spec_label
)

# IPAC points with error bars
plt.errorbar(
    wav_ipac,
    nuLnu_ipac.value,
    yerr=err_nuLnu_ipac.value,
    fmt="o",
    markersize=4,
    capsize=3,
    linestyle="none",
    label="SPHEREx"
)

if wise_wav_um.size > 0:
    plt.errorbar(
        wise_wav_um,
        nuLnu_wise.value,
        yerr=err_nuLnu_wise.value,
        fmt="s",
        markersize=6,
        capsize=3,
        linestyle="none",
        color="tab:red",
        label="WISE (AllWISE)"
    )
    for i, band in enumerate(wise_bands):
        plt.text(wise_wav_um[i] * 1.03, nuLnu_wise.value[i], band, color="tab:red", fontsize=9)

if sdss_phot_wav_used_um.size > 0:
    plt.errorbar(
        sdss_phot_wav_used_um,
        nuLnu_sdss_phot.value,
        yerr=err_nuLnu_sdss_phot.value,
        fmt="^",
        markersize=6,
        capsize=3,
        linestyle="none",
        color="tab:green",
        label="SDSS photometry"
    )
    for i, band in enumerate(sdss_phot_bands_used):
        plt.text(sdss_phot_wav_used_um[i] * 1.03, nuLnu_sdss_phot.value[i], band, color="tab:green", fontsize=9)

plt.xlabel("Wavelength (micron)")
plt.ylabel(r"$\nu L_\nu$ (erg s$^{-1}$)")
plt.xscale('log')
plt.yscale('log')
plt.title(f"{src_name}: {spec_source.upper()} + IPAC, z={z} ($\\nu L_\\nu$)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ------------------------------------------------------------
# Plot 3: nu*L_nu versus rest-frame wavelength, with line labels
# ------------------------------------------------------------
wav_um_rest_sdss = wav_um / (1.0 + z)
wav_um_rest_ipac = wav_ipac / (1.0 + z)
wav_um_rest_wise = wise_wav_um / (1.0 + z)
wav_um_rest_sdss_phot = sdss_phot_wav_used_um / (1.0 + z)

fig, (ax, ax_t) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)

# Optical spectrum
ax.plot(
    wav_um_rest_sdss[good_sdss],
    nuLnu_sdss.value[good_sdss],
    linewidth=1.0,
    label=spec_label
)

# IPAC points with error bars
ax.errorbar(
    wav_um_rest_ipac,
    nuLnu_ipac.value,
    yerr=err_nuLnu_ipac.value,
    fmt="o",
    markersize=4,
    capsize=3,
    linestyle="none",
    label="SPHEREx"
)

if wise_wav_um.size > 0:
    ax.errorbar(
        wav_um_rest_wise,
        nuLnu_wise.value,
        yerr=err_nuLnu_wise.value,
        fmt="s",
        markersize=6,
        capsize=3,
        linestyle="none",
        color="tab:red",
        label="WISE (AllWISE)"
    )

if sdss_phot_wav_used_um.size > 0:
    ax.errorbar(
        wav_um_rest_sdss_phot,
        nuLnu_sdss_phot.value,
        yerr=err_nuLnu_sdss_phot.value,
        fmt="^",
        markersize=6,
        capsize=3,
        linestyle="none",
        color="tab:green",
        label="SDSS photometry"
    )

# Key rest-frame emission lines (micron)
emission_lines = [
    ("Ly$\\alpha$", 0.1216),
    ("C IV", 0.1549),
    ("C III]", 0.1909),
    ("Mg II", 0.2798),
    ("[O II]", 0.3727),
    ("H$\\beta$", 0.4861),
    ("[O III]", 0.5007),
    ("H$\\alpha$", 0.6563),
    ("Pa$\\beta$", 1.282),
    ("Pa$\\alpha$", 1.875),
]

xmin_candidates = [wav_um_rest_sdss[good_sdss], wav_um_rest_ipac, wav_um_rest_wise, wav_um_rest_sdss_phot]
xmax_candidates = [wav_um_rest_sdss[good_sdss], wav_um_rest_ipac, wav_um_rest_wise, wav_um_rest_sdss_phot]
xmin = np.nanmin(np.concatenate([arr for arr in xmin_candidates if arr.size > 0]))
xmax = np.nanmax(np.concatenate([arr for arr in xmax_candidates if arr.size > 0]))
ymin, ymax = ax.get_ylim()
label_y = ymax / 1.5

for line_name, line_um in emission_lines:
    if xmin <= line_um <= xmax:
        ax.axvline(line_um, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            line_um * 1.01,
            label_y,
            line_name,
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color="gray"
        )

# Plot SDSS + WISE filter transmissions in a separate bottom panel
filter_ids = [
    "SLOAN/SDSS.u", "SLOAN/SDSS.g", "SLOAN/SDSS.r", "SLOAN/SDSS.i", "SLOAN/SDSS.z",
    "WISE/WISE.W1", "WISE/WISE.W2", "WISE/WISE.W3", "WISE/WISE.W4",
]
filter_colors = [
    "navy", "tab:green", "tab:orange", "tab:red", "tab:purple",
    "tab:gray", "tab:brown", "tab:olive", "tab:cyan",
]
plotted_filter = False

for filt_id, fcolor in zip(filter_ids, filter_colors):
    try:
        tcurve = SvoFps.get_transmission_data(filt_id)
        wav_obs_um = np.array(tcurve["Wavelength"], dtype=float) / 1e4  # Angstrom -> micron
        wav_rest_um = wav_obs_um / (1.0 + z)
        trans = np.array(tcurve["Transmission"], dtype=float)
        band = filt_id.split(".")[-1]
        ax_t.plot(
            wav_rest_um,
            trans,
            color=fcolor,
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
            label=f"{band} transmission"
        )
        plotted_filter = True
    except Exception as exc:
        print(f"Could not fetch {filt_id} transmission curve: {exc}")

if plotted_filter:
    ax_t.set_ylabel("Transmission")
    ax_t.set_ylim(0, 1.05)
    ax_t.legend(loc="upper right", fontsize=8)
else:
    ax_t.text(0.5, 0.5, "WISE transmission curves unavailable", ha="center", va="center", transform=ax_t.transAxes)

ax.set_ylabel(r"$\nu L_\nu$ (erg s$^{-1}$)")
ax.set_xscale('log')
# ax.set_yscale('log')
ax_t.set_xscale('log')
ax_t.set_xlabel("Rest-frame Wavelength (micron)")
ax.set_title(f"{src_name}: Rest-frame, z={z} ($\\nu L_\\nu$)")
ax.grid(True, alpha=0.3)
ax_t.grid(True, alpha=0.3)

ax.legend(loc="best")
fig.tight_layout()
last_fig_name = f"{src_name}_restframe_nuLnu.png"
fig.savefig(last_fig_name, dpi=300, bbox_inches="tight")
print(f"Saved figure: {last_fig_name}")
plt.show()
