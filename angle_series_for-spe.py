import os
import re
import struct
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from pathlib import Path

# adjust this path if necessary or prompt user
DATA_FOLDER = r"C:\Users\julia\Desktop\20260727_p01\1518\Polarization\Excitation_Sweep"

# eV*nm, times 1000 for meV
_EV_NM = 1239.84193

# --- SPE binary format (Princeton Instruments WinSpec/LightField 2.x/3.x) ---
# 4100-byte fixed header, little-endian.
_HEADER_SIZE = 4100
_OFF_XDIM = 42        # uint16 - pixels per spectrum
_OFF_DATATYPE = 108   # uint16 - see _DATATYPE_TO_NUMPY
_OFF_YDIM = 656        # uint16 - typically 1 for spectra
_OFF_NUM_FRAMES = 1446  # uint32
_OFF_VERSION = 1992    # float32 - 2.0 / 3.0
_OFF_XML_FOOTER = 678   # int64 - absolute byte-offset of SPE 3.x XML footer
# XCalibration block: 480 bytes starting at offset 3000.
_OFF_XCAL = 3000
_OFF_XCAL_OFFSET = _OFF_XCAL + 0        # float64
_OFF_XCAL_FACTOR = _OFF_XCAL + 8        # float64
_OFF_XCAL_POLY_ORDER = _OFF_XCAL + 18   # uint8
_OFF_XCAL_POLY_COEFFS = _OFF_XCAL + 263  # float64[6]
_N_POLY_COEFFS = 6

_DATATYPE_TO_NUMPY = {
    0: np.dtype("<f4"),
    1: np.dtype("<i4"),
    2: np.dtype("<i2"),
    3: np.dtype("<u2"),
    5: np.dtype("<f8"),
    6: np.dtype("<u1"),
    8: np.dtype("<u4"),
}


def _localname(tag):
    """Strip the {namespace} prefix ElementTree puts on tags."""
    return tag.rsplit('}', 1)[-1]


def _as_positive_float(text):
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _parse_xml_footer_wavelength(raw, xdim):
    """Parse the SPE 3.x trailing XML footer for a per-pixel nm axis.

    Returns wavelength_nm_array or None if no usable footer wavelength
    exists; falls back to the legacy XCalibration path in that case.
    """
    try:
        if len(raw) < _OFF_XML_FOOTER + 8:
            return None
        footer_off = struct.unpack_from("<q", raw, _OFF_XML_FOOTER)[0]
        if not (0 < footer_off < len(raw)):
            return None
        xml_text = raw[footer_off:].decode("utf-8", errors="replace")
        root = ET.fromstring(xml_text)

        wavelength_axis = None
        for el in root.iter():
            if _localname(el.tag) == "Wavelength" and wavelength_axis is None:
                text = (el.text or "").strip()
                if not text:
                    continue
                try:
                    values = [float(p) for p in text.split(",")]
                except ValueError:
                    continue
                if len(values) == xdim:
                    wavelength_axis = np.asarray(values, dtype=np.float64)
        return wavelength_axis
    except Exception:
        return None


def _evaluate_calibration(offset, factor, poly_order, poly_coeffs, pixel_index):
    """Evaluate the SPE XCalibration block for a vector of pixel indices."""
    coeffs = list(poly_coeffs)
    if poly_order >= 1 and any(abs(c) > 0.0 for c in coeffs[:poly_order + 1]):
        wn = np.zeros_like(pixel_index, dtype=np.float64)
        for power, c in enumerate(coeffs[:poly_order + 1]):
            wn += c * (pixel_index ** power)
        return wn
    if factor != 0.0:
        return offset + factor * pixel_index
    return pixel_index.copy()


def load_spe_spectrum(filepath):
    """Load a spectrum from a Princeton Instruments .spe file.

    Returns (energies_mev, intensity), sorted ascending by energy.
    """
    raw = Path(filepath).read_bytes()
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"{filepath}: file is too short to be a valid SPE file")

    header = raw[:_HEADER_SIZE]
    xdim = struct.unpack_from("<H", header, _OFF_XDIM)[0]
    ydim = max(struct.unpack_from("<H", header, _OFF_YDIM)[0], 1)
    datatype = struct.unpack_from("<H", header, _OFF_DATATYPE)[0]
    num_frames = max(struct.unpack_from("<I", header, _OFF_NUM_FRAMES)[0], 1)
    version_value = struct.unpack_from("<f", header, _OFF_VERSION)[0]
    xcal_offset = struct.unpack_from("<d", header, _OFF_XCAL_OFFSET)[0]
    xcal_factor = struct.unpack_from("<d", header, _OFF_XCAL_FACTOR)[0]
    xcal_poly_order = struct.unpack_from("<B", header, _OFF_XCAL_POLY_ORDER)[0]
    xcal_poly_coeffs = struct.unpack_from(f"<{_N_POLY_COEFFS}d", header, _OFF_XCAL_POLY_COEFFS)

    if datatype not in _DATATYPE_TO_NUMPY:
        raise ValueError(f"{filepath}: unsupported SPE datatype code {datatype}")
    dtype = _DATATYPE_TO_NUMPY[datatype]

    expected_data_bytes = xdim * ydim * num_frames * dtype.itemsize
    data_blob = raw[_HEADER_SIZE:_HEADER_SIZE + expected_data_bytes]
    if len(data_blob) != expected_data_bytes:
        raise ValueError(f"{filepath}: corrupt or truncated SPE file")

    cube = np.frombuffer(data_blob, dtype=dtype).reshape(num_frames, ydim, xdim)
    summed = cube.sum(axis=0).astype(np.float64)
    intensity = summed[0] if ydim == 1 else summed.sum(axis=0)

    spe_version = 3 if 2.5 <= float(version_value) < 4.0 else 2

    pixel_index = np.arange(xdim, dtype=np.float64)
    wavelength_nm = None
    if spe_version >= 3:
        wavelength_nm = _parse_xml_footer_wavelength(raw, xdim)
    if wavelength_nm is None:
        wavelength_nm = _evaluate_calibration(
            xcal_offset, xcal_factor, xcal_poly_order, xcal_poly_coeffs, pixel_index
        )

    energies_mev = 1000.0 * _EV_NM / wavelength_nm

    # ascending wavelength -> descending energy; flip to ascending energy
    order = np.argsort(energies_mev)
    return energies_mev[order], intensity[order]


def list_data_files(folder, include_raw=False):
    """Return sorted list of .spe files in folder, excluding '-raw.spe' pairs by default."""
    files = [f for f in os.listdir(folder) if f.lower().endswith('.spe')]
    if not include_raw:
        files = [f for f in files if not f.lower().endswith('-raw.spe')]
    return sorted(files)


def extract_angle(filename):
    """Extract numeric angle from filename using regex."""
    m = re.search(r'([-+]?[0-9]*\.?[0-9]+)', filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def differentiate_spectrum(energies, intensity):
    """Compute the derivative of intensity with respect to energy."""
    return np.gradient(intensity, energies)


def integrate_interval(energies, intensity, start, end):
    """Integrate intensity over given energy interval (meV) using trapezoidal rule."""
    mask = (energies >= start) & (energies <= end)
    if not np.any(mask):
        return 0.0
    return np.trapezoid(intensity[mask], energies[mask])


def main():
    parser = argparse.ArgumentParser(
        description="Load SPE spectra, integrate over a selected energy interval, and display the angle dependence on a polar plot."
    )
    parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, help="path to folder containing .spe spectra")
    parser.add_argument("-b", "--background", type=float, default=0.0, help="constant background intensity to subtract from each spectrum")
    parser.add_argument("-normalize", "--normalize", action="store_true", help="normalize integrated intensities by the maximum value")
    parser.add_argument("--differentiate", action="store_true", help="use differentiated spectra for integration")
    parser.add_argument("-s", "--stepsize", type=float, default=None, help="manual angle stepsize (degrees) between successive files, in sorted file order; overrides filename-based angle extraction")
    parser.add_argument("--include-raw", action="store_true", help="also include '*-raw.spe' files (excluded by default)")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    background = args.background
    normalize = args.normalize

    if not folder.exists():
        print(f"Data folder does not exist: {folder}")
        return

    # Create output folder
    input_folder_name = folder.name
    output_folder = folder.parent / f"results_{input_folder_name}"
    output_folder.mkdir(parents=True, exist_ok=True)

    files = list_data_files(str(folder), include_raw=args.include_raw)
    if not files:
        print(f"No .spe files found in {folder}")
        return

    print("Available spectra:")
    for i, fname in enumerate(files, start=1):
        print(f" {i}. {fname}")

    # Load all spectra, subtract background, and assign angles
    all_spectra = {}
    angles = []
    for i, fname in enumerate(files):
        energies, inten = load_spe_spectrum(str(folder / fname))
        if background != 0.0:
            inten = inten - background
        all_spectra[fname] = (energies, inten)
        if args.stepsize is not None:
            angles.append(2 * i * args.stepsize)
        else:
            angles.append(2 * extract_angle(fname))

    # Use first spectrum for interactive selection
    first_file = files[0]
    energies_ref, intensity_ref = all_spectra[first_file]

    print(f"\nUsing spectrum: {first_file}")
    print("Select an energy interval on the spectrum by clicking and dragging.")

    # Create figure with three subplots
    fig = plt.figure(figsize=(21, 6))

    # Left: spectrum with span selector
    ax_spectrum = fig.add_subplot(131)
    ax_spectrum.plot(energies_ref, intensity_ref, 'b-')
    ax_spectrum.set_xlim(np.nanmin(energies_ref), np.nanmax(energies_ref))
    ax_spectrum.set_xlabel('Energy (meV)')
    ax_spectrum.set_ylabel('Intensity')
    ax_spectrum.set_title(f'Select Integration Interval\n({first_file})')
    ax_spectrum.grid(True, alpha=0.3)

    # Middle: differentiated spectrum
    ax_diff = fig.add_subplot(132)
    d_inten_ref = differentiate_spectrum(energies_ref, intensity_ref)
    ax_diff.plot(energies_ref, d_inten_ref, 'r-')
    ax_diff.set_xlim(np.nanmin(energies_ref), np.nanmax(energies_ref))
    ax_diff.set_xlabel('Energy (meV)')
    ax_diff.set_ylabel('Differentiated Intensity')
    ax_diff.set_title(f'Differentiated Spectrum\n({first_file})')
    ax_diff.grid(True, alpha=0.3)

    # Right: polar plot
    ax_polar = fig.add_subplot(133, projection='polar')

    # Initialize span selector state
    state = {'start': None, 'end': None, 'polar_line': None, 'angles_sorted': None, 'norm_sorted': None}

    def on_select(xmin, xmax):
        """Callback when user selects span on spectrum."""
        state['start'] = xmin
        state['end'] = xmax

        # Integrate all spectra over selected interval
        integrated = []
        for fname in files:
            energies, inten = all_spectra[fname]
            if args.differentiate:
                d_inten = differentiate_spectrum(energies, inten)
                val = integrate_interval(energies, d_inten, xmin, xmax)
            else:
                val = integrate_interval(energies, inten, xmin, xmax)
            integrated.append(val)

        integrated = np.array(integrated)
        integrated = np.abs(integrated)

        # Normalize if requested
        if normalize:
            max_integrated = integrated.max()
            if integrated.size and max_integrated > 0:
                data_to_plot = integrated / max_integrated
            else:
                data_to_plot = np.zeros_like(integrated)
        else:
            data_to_plot = integrated

        # Sort by angle for plotting
        angles_array = np.array(angles)
        order = np.argsort(angles_array)
        angles_sorted = angles_array[order]
        norm_sorted = data_to_plot[order]

        state['angles_sorted'] = angles_sorted
        state['norm_sorted'] = norm_sorted

        # Convert to radians
        theta = np.deg2rad(angles_sorted)

        # Update polar plot
        ax_polar.clear()
        ax_polar.plot(theta, norm_sorted, marker='o', markersize=6)
        plot_title = "Differentiated" if args.differentiate else "Raw"
        intensity_label = "Normalized Intensity" if normalize else "Integrated Intensity"
        ax_polar.set_title(f"{plot_title} {intensity_label}\n({xmin:.2f}-{xmax:.2f} meV)")
        fig.canvas.draw_idle()

    # Create span selector on spectrum
    span = SpanSelector(
        ax_spectrum,
        on_select,
        direction='horizontal',
        props=dict(alpha=0.3, facecolor='red'),
        interactive=True
    )

    plt.tight_layout()
    plt.show()

    # Save result if interval was selected
    if state['start'] is not None and state['end'] is not None:
        start, end = state['start'], state['end']
        plot_title = "differentiated" if args.differentiate else "raw"
        save_path = output_folder / f"polar_{plot_title}_{int(start)}_{int(end)}meV.png"
        fig.savefig(save_path, dpi=150)
        print(f"Polar plot saved to {save_path}")

        # Save data to CSV
        csv_path = output_folder / f"polar_{plot_title}_{int(start)}_{int(end)}meV.csv"
        if state['angles_sorted'] is not None and state['norm_sorted'] is not None:
            data_to_save = np.column_stack((state['angles_sorted'], state['norm_sorted']))
            csv_header = 'Angle (degrees),Normalized Intensity' if normalize else 'Angle (degrees),Integrated Intensity'
            np.savetxt(csv_path, data_to_save, delimiter=',', header=csv_header, comments='')
            print(f"Polar plot data saved to {csv_path}")

if __name__ == '__main__':
    main()
