import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from pathlib import Path

# adjust this path if necessary or prompt user
DATA_FOLDER = r"C:\Users\julia\Desktop\20260727_p01\1518\Polarization"

def list_data_files(folder):
    """Return sorted list of .dat files in folder."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.dat')])
    return files

def load_spectrum(filepath):
    """Load two-column spectrum from a .dat file (energy [meV], intensity)."""
    try:
        data = np.loadtxt(filepath, skiprows=1)
    except ValueError:
        data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]

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

def calculate_degree_of_polarization(intensities):
    """Calculate degree of polarization from intensity data."""
    if len(intensities) == 0 or np.max(intensities) + np.min(intensities) == 0:
        return 0.0
    i_max = np.max(intensities)
    i_min = np.min(intensities)
    dop = (i_max - i_min) / (i_max + i_min)
    return dop

def main():
    parser = argparse.ArgumentParser(
        description="Load spectra, integrate over a selected energy interval, and display the angle dependence on a polar plot."
    )
    parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, help="path to folder containing .dat spectra")
    parser.add_argument("-b", "--background", type=float, default=0.0, help="constant background intensity to subtract from each spectrum")
    parser.add_argument("-normalize", "--normalize", action="store_true", help="normalize integrated intensities by the maximum value")
    parser.add_argument("--differentiate", action="store_true", help="use differentiated spectra for integration")
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

    files = list_data_files(str(folder))
    if not files:
        print(f"No .dat files found in {folder}")
        return

    print("Available spectra:")
    for i, fname in enumerate(files, start=1):
        print(f" {i}. {fname}")

    # Load all spectra, subtract background, and extract angles
    all_spectra = {}
    angles = []
    for fname in files:
        energies, inten = load_spectrum(str(folder / fname))
        if background != 0.0:
            inten = inten - background
        all_spectra[fname] = (energies, inten)
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
    state = {'start': None, 'end': None, 'polar_line': None, 'angles_sorted': None, 'norm_sorted': None, 'dop': None}

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

        # Calculate degree of polarization
        dop = calculate_degree_of_polarization(norm_sorted)
        state['dop'] = dop

        # Convert to radians
        theta = np.deg2rad(angles_sorted)

        # Update polar plot
        ax_polar.clear()
        ax_polar.plot(theta, norm_sorted, marker='o', markersize=6)
        plot_title = "Differentiated" if args.differentiate else "Raw"
        intensity_label = "Normalized Intensity" if normalize else "Integrated Intensity"
        ax_polar.set_title(f"{plot_title} {intensity_label}\n({xmin:.2f}-{xmax:.2f} meV)\nDOP: {dop:.4f}")
        fig.canvas.draw_idle()

        # Print to console
        print(f"\nEnergy interval: {xmin:.2f}-{xmax:.2f} meV")
        print(f"Degree of Polarization (DOP): {dop:.4f}")
        print(f"Max intensity: {np.max(norm_sorted):.6f}, Min intensity: {np.min(norm_sorted):.6f}")

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

        # Print DOP summary
        if state['dop'] is not None:
            print(f"\nSummary:")
            print(f"  Degree of Polarization (DOP): {state['dop']:.4f}")

if __name__ == '__main__':
    main()