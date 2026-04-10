# ============================================================
# RAMAN SPECTROSCOPY ANALYZER
# Texas Instruments Python Calculator Script
# Compatible: TI-84 Plus CE Python, TI-Nspire CX II
#
# No external libraries needed — pure Python + ti_plotlib.
# Falls back to ASCII output on a standard PC.
#
# Usage on calc: Open in TI-84/Nspire editor and press Run.
# Usage on PC  : python ti_raman.py
# ============================================================

import math

try:
    import ti_plotlib as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── Spectral window & resolution ────────────────────────────
WN_MIN   = 100    # cm-1 start
WN_MAX   = 3300   # cm-1 end
N_PTS    = 60     # background grid points (peaks added separately)

# ── Material peak catalogue: (center cm-1, HWHM, amplitude) ─
# Source: RRUFF database / standard Raman references
MATERIALS = {
    "1-Silicon":  [
        (521,  4,  1.00),          # Si-Si optical phonon
    ],
    "2-Diamond":  [
        (1332, 6,  1.00),          # sp3 C-C stretch
    ],
    "3-Graphene": [
        (1350, 25, 0.35),          # D band (defects)
        (1580, 18, 1.00),          # G band (sp2 C)
        (2700, 35, 0.65),          # 2D band
    ],
    "4-TiO2":     [
        (144,  6,  1.00),          # Eg mode (anatase)
        (399,  12, 0.35),          # B1g mode
        (513,  10, 0.25),          # A1g+B1g
        (639,  12, 0.45),          # Eg mode
    ],
    "5-Calcite":  [
        (280,  10, 0.30),          # lattice mode
        (712,  8,  0.25),          # CO3 bending
        (1085, 7,  1.00),          # CO3 sym stretch
    ],
    "6-Quartz":   [
        (128,  8,  0.60),          # lattice
        (206,  10, 0.50),          # lattice
        (464,  10, 1.00),          # Si-O-Si sym stretch
    ],
    "7-Polysty":  [
        (1001, 4,  1.00),          # ring breathing
        (1031, 5,  0.40),          # C-H deformation
        (1583, 8,  0.30),          # C=C ring stretch
        (3054, 8,  0.55),          # aromatic C-H
    ],
    "8-PMMA":     [
        (813,  6,  0.45),          # O-CH3 rocking
        (1452, 8,  0.50),          # CH3/CH2 deform
        (1727, 8,  1.00),          # C=O ester stretch
        (2953, 10, 0.70),          # C-H stretch
    ],
}

# ── Color map for ti_plotlib ─────────────────────────────────
COLORS = [
    (0,   120, 255),   # blue
    (255, 60,  60),    # red
    (0,   180, 80),    # green
    (255, 160, 0),     # orange
    (160, 0,   255),   # purple
]

# ── Core math ────────────────────────────────────────────────
def lorentzian(x, c, w, a):
    """Lorentzian (Cauchy) peak: I = a / (1 + ((x-c)/w)^2)"""
    return a / (1.0 + ((x - c) / w) ** 2)

def linspace(start, stop, n):
    """Pure-Python linspace (no numpy)."""
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]

def make_spectrum(peaks):
    """Return (wavenumbers, intensities) sorted lists.

    Strategy: combine a coarse background grid with dense points around
    each peak center (±3×HWHM, 12 pts each).  This ensures narrow peaks
    like Silicon at 521 cm-1 are captured accurately regardless of grid
    spacing, while keeping the total point count low for TI memory.
    """
    # Background grid
    wn_set = set()
    bg = linspace(WN_MIN, WN_MAX, N_PTS)
    for x in bg:
        wn_set.add(round(x, 2))

    # Dense points around each peak (±3×HWHM, 12 pts)
    for (c, w, a) in peaks:
        band = linspace(c - 3 * w, c + 3 * w, 12)
        for x in band:
            if WN_MIN <= x <= WN_MAX:
                wn_set.add(round(x, 2))

    wn = sorted(wn_set)
    intensity = []
    for x in wn:
        val = 0.0
        for (c, w, a) in peaks:
            val += lorentzian(x, c, w, a)
        intensity.append(val)

    return wn, intensity

def normalize(intensity):
    """Min-max normalize to [0, 1]."""
    mn = min(intensity)
    mx = max(intensity)
    rng = mx - mn if mx != mn else 1.0
    return [(v - mn) / rng for v in intensity], mx

def peak_summary(peaks):
    """Return list of (center, rel_amp) sorted by amplitude desc."""
    mx = max(a for (_, _, a) in peaks)
    return sorted([(c, a / mx) for (c, _, a) in peaks],
                  key=lambda x: -x[1])

# ── Display helpers ──────────────────────────────────────────
def print_header(title):
    print("=" * 24)
    print(title)
    print("=" * 24)

def print_peaks(peaks):
    print("Peak positions (cm-1):")
    for c, _, a in sorted(peaks, key=lambda x: x[0]):
        bar = int(a * 10)
        print(f"  {c:5d} |{'#' * bar}  {a:.2f}")

def ascii_spectrum(wn, norm_intensity):
    """Print a compact ASCII line plot (40-char wide)."""
    W = 38
    print(f"\n{'~' * (W + 2)}")
    print(f" {WN_MIN}{'cm-1':>{W - 4}}{WN_MAX}")
    N = len(norm_intensity)
    step = max(1, N // W)
    row = ""
    for i in range(0, N, step):
        v = norm_intensity[i]
        row += "#" if v > 0.60 else ("." if v > 0.20 else " ")
    print("|" + row[:W].ljust(W) + "|")
    print(f"{'~' * (W + 2)}\n")

def ti_plot_spectrum(name, wn, intensity):
    """Draw the spectrum using ti_plotlib."""
    norm, mx = normalize(intensity)
    plt.cls()
    plt.window(WN_MIN, WN_MAX, 0, 1.15)
    plt.axes("on")
    plt.color(0, 120, 255)
    plt.plot(wn, norm, "", "blue")
    # Mark individual peaks
    plt.color(255, 60, 60)
    for c, _, a in MATERIALS[name]:
        y_norm = (a / mx)
        plt.scatter([c], [min(y_norm, 1.0)], ".", "red")
    plt.title(name.split("-", 1)[1], 0.05, 1.08, "black")
    plt.show()

# ── Comparison: overlay two spectra ─────────────────────────
def ti_compare(name_a, wn_a, ia, name_b, wn_b, ib):
    """Overlay two normalized spectra in different colors."""
    na, _ = normalize(ia)
    nb, _ = normalize(ib)
    plt.cls()
    plt.window(WN_MIN, WN_MAX, 0, 1.3)
    plt.axes("on")
    plt.color(0, 120, 255)
    plt.plot(wn_a, na, "", "blue")
    plt.color(255, 60, 60)
    plt.plot(wn_b, nb, "", "red")
    lbl = name_a.split("-", 1)[1] + " vs " + name_b.split("-", 1)[1]
    plt.title(lbl, 0.02, 1.22, "black")
    plt.show()

# ── Main menu ────────────────────────────────────────────────
def main():
    print_header("RAMAN ANALYZER")
    keys = list(MATERIALS.keys())

    while True:
        print("\n-- Materials --")
        for k in keys:
            print(" " + k)
        print(" 9-Compare two")
        print(" 0-Quit")

        choice = input("Select: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "9":
            # Compare mode
            a = input("First  (1-8): ").strip()
            b = input("Second (1-8): ").strip()
            ka = next((k for k in keys if k.startswith(a)), None)
            kb = next((k for k in keys if k.startswith(b)), None)
            if ka is None or kb is None:
                print("Invalid selection")
                continue
            wn_a, ia = make_spectrum(MATERIALS[ka])
            wn_b, ib = make_spectrum(MATERIALS[kb])
            print_header(ka.split("-",1)[1] + " vs " + kb.split("-",1)[1])
            print("\n--- " + ka.split("-",1)[1] + " ---")
            print_peaks(MATERIALS[ka])
            print("\n--- " + kb.split("-",1)[1] + " ---")
            print_peaks(MATERIALS[kb])
            if HAS_PLOT:
                ti_compare(ka, wn_a, ia, kb, wn_b, ib)
            else:
                na, _ = normalize(ia)
                ascii_spectrum(wn_a, na)

        else:
            mat = next((k for k in keys if k.startswith(choice)), None)
            if mat is None:
                print("Invalid selection")
                continue

            peaks = MATERIALS[mat]
            wn, intensity = make_spectrum(peaks)
            norm, mx = normalize(intensity)

            # Find strongest peak position
            i_max = intensity.index(max(intensity))
            strongest = wn[i_max]

            print_header(mat.split("-", 1)[1])
            print_peaks(peaks)
            print(f"\nStrongest: {strongest:.0f} cm-1")
            print(f"Max amp:   {mx:.3f}")

            top = peak_summary(peaks)
            print("\nRel. peak heights:")
            for c, r in top:
                print(f"  {c:5d} cm-1 : {r:.2f}")

            if HAS_PLOT:
                ti_plot_spectrum(mat, wn, intensity)
            else:
                ascii_spectrum(wn, norm)

            again = input("\nAnother? (y/n): ").strip().lower()
            if again != "y":
                print("Bye!")
                break

main()
