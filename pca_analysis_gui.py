#!/usr/bin/env python3
"""
PCA Analysis GUI Application
═════════════════════════════════════════════════════════════════════════════
A full-featured desktop application for Principal Component Analysis (PCA)
and interactive Raman spectroscopy exploration.

Capabilities
────────────
• Load tabular datasets (sklearn built-ins, arbitrary CSV, or Raman CSV).
• Standardise features and split into train / test subsets.
• Run PCA and display: scatter plot, scree / cumulative-variance, decision
  boundaries, and a component-loading heatmap.
• Optional Logistic Regression classifier evaluated on the PCA-projected data.
• Raman viewer: synthetic material presets or user-supplied CSV, with a live
  crosshair, per-spectrum normalisation, and stacked-offset display.

Dependencies
────────────
    pip install numpy pandas matplotlib scikit-learn
    # tkinter ships with CPython on Windows and macOS.
    # Linux: sudo apt install python3-tk

CSV formats supported
─────────────────────
General  : last column = class label; all other columns = numeric features.
Raman A  : first column = wavenumber axis; remaining columns = one spectrum
           each (column header used as sample label).
Raman B  : first column = sample label; column headers = wavenumber values.
"""

# ── Standard-library imports ──────────────────────────────────────────────────
import tkinter as tk                                          # Core GUI toolkit
from tkinter import ttk, filedialog, messagebox, scrolledtext # Themed widgets + dialogs
import io                                                     # StringIO for text buffering
import threading                                              # Background thread for PCA

# ── Third-party: numerical / data ─────────────────────────────────────────────
import numpy as np    # Array maths and random-number generation
import pandas as pd   # DataFrame for tabular data and CSV I/O

# ── Third-party: matplotlib ───────────────────────────────────────────────────
import matplotlib                                             # Top-level package
matplotlib.use("TkAgg")                                       # Must be set before pyplot import
import matplotlib.pyplot as plt                               # Pyplot API (used sparingly here)
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,      # Renders a Figure inside a Tk widget
    NavigationToolbar2Tk,   # Zoom / pan toolbar that integrates with Tk
)
from matplotlib.colors import ListedColormap  # Discrete colormap for decision-boundary fills
from matplotlib.figure import Figure          # OO Figure container (preferred over plt.figure)

# ── Third-party: scikit-learn ─────────────────────────────────────────────────
from sklearn.decomposition import PCA                        # Core PCA algorithm
from sklearn.preprocessing import StandardScaler             # Zero-mean / unit-variance scaling
from sklearn.model_selection import train_test_split         # Stratified train/test split
from sklearn.linear_model import LogisticRegression          # Classifier for boundary visualisation
from sklearn.metrics import confusion_matrix, accuracy_score # Classifier evaluation
from sklearn.datasets import load_iris, load_wine, load_breast_cancer  # Bundled sample datasets

# ── Optional: scipy for advanced signal-processing filters ────────────────────
try:
    from scipy.signal import savgol_filter as _savgol_filter    # Savitzky-Golay polynomial smoother
    from scipy.ndimage import gaussian_filter1d as _gfilt1d     # Gaussian convolution smoother
    from scipy.signal import medfilt as _medfilt                # Median filter for spike removal
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False   # scipy not installed; numpy-only fallbacks will be used


# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette  (dark GitHub-inspired theme)
# ═══════════════════════════════════════════════════════════════════════════════
BG           = "#0f1117"   # Deepest background — main window canvas
BG_SECONDARY = "#161b22"   # Slightly lighter surface — toolbar, text-box backgrounds
BG_CARD      = "#1c2333"   # Card / panel surface — axes background, notebook tab body
BORDER       = "#2a3040"   # Subtle separator lines and spine colour
TEXT         = "#e6edf3"   # Primary readable text
TEXT_DIM     = "#7d8590"   # De-emphasised text — labels, axis tick marks
ACCENT       = "#58a6ff"   # Interactive highlight — active tab, run button, crosshair
ACCENT_HOVER = "#79c0ff"   # Lighter accent for button hover state
GREEN        = "#3fb950"   # Positive / success indicator (cumulative variance fill)
RED          = "#f85149"   # Warning / threshold marker (95 % variance line)
ORANGE       = "#d29922"   # Secondary data series on variance plot
PURPLE       = "#bc8cff"   # Unused reserved colour
CYAN         = "#39d2c0"   # Monospace statistics text

# Six distinct colours cycled across class scatter points and Raman traces
PLOT_COLORS = ["#ff6b6b", "#51cf66", "#339af0", "#fcc419", "#cc5de8", "#22b8cf"]

# Two discrete colormaps used as semi-transparent fills in the decision-boundary panels
PLOT_CMAPS = [
    ListedColormap(["#ffd43b", "#f8f9fa", "#96f2d7"]),  # Warm palette — training panel
    ListedColormap(["#ffa8a8", "#d0ebff", "#b2f2bb"]),  # Cool palette — test panel
]


# ═══════════════════════════════════════════════════════════════════════════════
# Raman spectroscopy helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Shared wavenumber axis for all synthetic Raman spectra (cm⁻¹).
# 650 evenly-spaced points from 100 to 3300 cm⁻¹ gives ~4.9 cm⁻¹ resolution.
RAMAN_WN = np.linspace(100, 3300, 650)

# Peak catalogue: maps material name → list of (center, half-width, amplitude) tuples.
# All three values are in cm⁻¹ / a.u. units respectively.
# Sources: standard Raman reference databases (RRUFF, literature).
RAMAN_MATERIALS = {
    # ── Semiconductors / Carbon allotropes ────────────────────────────────────
    "Silicon": [
        (521,  4,  1.00),   # First-order Si-Si optical phonon mode
    ],
    "Diamond": [
        (1332, 6,  1.00),   # Zone-centre sp³ C-C stretching mode
    ],
    "Graphene": [
        (1350, 25, 0.35),   # D band — activated by defects (inter-valley scattering)
        (1580, 18, 1.00),   # G band — in-plane E₂g stretching of sp² carbons
        (2700, 35, 0.65),   # 2D band — second-order overtone of the D band
    ],
    # ── Minerals ──────────────────────────────────────────────────────────────
    "TiO2 (Anatase)": [
        (144,  6,  1.00),   # Eg mode — strongest peak, characteristic of anatase
        (197,  8,  0.15),   # Eg mode — weak shoulder
        (399,  12, 0.35),   # B1g mode
        (513,  10, 0.25),   # A1g + B1g overlap
        (639,  12, 0.45),   # Eg mode
    ],
    "Calcite": [
        (280,  10, 0.30),   # Lattice vibration (translational mode)
        (712,  8,  0.25),   # In-plane bending ν₄ of CO₃²⁻
        (1085, 7,  1.00),   # Symmetric stretching ν₁ of CO₃²⁻ — dominant peak
    ],
    "Quartz": [
        (128,  8,  0.60),   # Lattice mode
        (206,  10, 0.50),   # Lattice mode
        (464,  10, 1.00),   # Si-O-Si symmetric stretching — strongest peak
        (698,  8,  0.15),   # Si-O-Si bending
        (795,  8,  0.20),   # Si-O stretching
        (1082, 12, 0.15),   # Asymmetric Si-O-Si stretching
    ],
    "Gypsum": [
        (415,  8,  0.40),   # SO₄²⁻ ν₂ bending
        (492,  8,  0.55),   # SO₄²⁻ ν₂ bending
        (1008, 7,  1.00),   # SO₄²⁻ ν₁ symmetric stretching — dominant
        (1140, 6,  0.30),   # SO₄²⁻ ν₃ asymmetric stretching
        (3405, 12, 0.50),   # O-H stretching of crystal water
    ],
    # ── Polymers ──────────────────────────────────────────────────────────────
    "Polystyrene": [
        (621,  5,  0.25),   # Ring deformation
        (1001, 4,  1.00),   # Ring breathing mode — diagnostic marker
        (1031, 5,  0.40),   # C-H in-plane deformation
        (1155, 5,  0.15),   # C-C stretching
        (1583, 8,  0.30),   # Ring C=C stretching
        (1602, 6,  0.40),   # Ring C=C stretching (split component)
        (3054, 8,  0.55),   # Aromatic C-H stretching
    ],
    "PMMA": [
        (813,  6,  0.45),   # O-CH₃ rocking + C-O-C stretching — diagnostic
        (987,  6,  0.30),   # C-O-C stretching
        (1452, 8,  0.50),   # CH₃ / CH₂ deformation
        (1727, 8,  1.00),   # C=O ester stretching — strongest peak
        (2953, 10, 0.70),   # C-H stretching
    ],
    "Polyethylene": [
        (1063, 8,  0.80),   # C-C stretching (all-trans)
        (1130, 8,  1.00),   # C-C stretching (all-trans) — dominant
        (1296, 6,  0.45),   # CH₂ twisting
        (2848, 10, 0.90),   # CH₂ symmetric stretching
        (2883, 10, 1.00),   # CH₂ asymmetric stretching
    ],
}

# Maps preset display name → list of (material, n_spectra) pairs.
# Used to populate the Raman preset combobox and generate synthetic datasets.
RAMAN_PRESETS = {
    "Raman: Semiconductors (Si / Diamond / Graphene)": [
        ("Silicon", 20), ("Diamond", 20), ("Graphene", 20),
    ],
    "Raman: Minerals (Calcite / Quartz / TiO2)": [
        ("Calcite", 20), ("Quartz", 20), ("TiO2 (Anatase)", 20),
    ],
    "Raman: Polymers (Polystyrene / PMMA / PE)": [
        ("Polystyrene", 20), ("PMMA", 20), ("Polyethylene", 20),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Coating simulation — peak catalogues
# ═══════════════════════════════════════════════════════════════════════════════
# Each coating is defined as a list of (center, half-width, amplitude) Lorentzian
# peaks that represent the coating's own Raman signature.  When the coating is
# applied to a spectrum the substrate signal is attenuated and the coating
# peaks are added on top, modulated by the user-controlled thickness factor.
#
# References:
#   InCrAlC — PVD hard coatings, DLC/CrC Raman: Sánchez-López et al. (2004)
#   Silane   — PDMS / organosilane: Smith & Dent, "Modern Raman Spectroscopy" (2005)
#   Paraffin — long-chain alkane: Snyder et al., J. Chem. Phys. (1961)
#   Carnauba — long-chain ester wax: Iwata et al., Lipids (1997)

COATING_CATALOGUE = {
    # ── InCrAlC  (PVD hard coating: DLC carbon + CrC + AlN components) ────────
    "InCrAlC": [
        (580,  20, 0.25),   # Cr-C stretching vibration
        (665,  25, 0.15),   # AlN acoustic-branch mode
        (1360, 80, 0.45),   # D band — disordered carbon (sp³/sp² boundary)
        (1560, 60, 0.65),   # G band — graphitic sp² carbon (red-shifted vs pure graphene)
    ],
    # ── Silane  (organosilane / PDMS coupling layer) ───────────────────────────
    "Silane": [
        (450,  30, 0.35),   # Si-O-Si bending (δ)
        (710,  15, 0.20),   # Si-CH₃ rocking
        (790,  12, 0.45),   # Si-C stretching + Si-O-Si symmetric stretch
        (1025, 20, 0.30),   # Si-O-Si asymmetric stretching
        (1265, 10, 0.15),   # Si-CH₃ symmetric deformation
        (2905, 15, 0.25),   # CH₃ C-H symmetric stretching
        (2965, 12, 0.20),   # CH₃ C-H asymmetric stretching
    ],
    # ── Paraffin wax  (straight-chain alkane, similar to PE) ──────────────────
    "Wax (Paraffin)": [
        (1063, 8,  0.55),   # C-C stretching (all-trans chains)
        (1130, 8,  0.70),   # C-C stretching (all-trans chains)
        (1170, 6,  0.20),   # C-C stretching + CH₂ wagging
        (1296, 6,  0.40),   # CH₂ twisting
        (1440, 8,  0.45),   # CH₂ scissoring (δ)
        (2848, 10, 0.85),   # CH₂ symmetric stretching (ν_s)
        (2883, 10, 0.95),   # CH₂ asymmetric stretching (ν_as)
        (2920, 10, 0.50),   # CH₂ asymmetric stretching (mobile/liquid-like chains)
    ],
    # ── Carnauba wax  (long-chain fatty ester; harder, higher-melting than paraffin)
    "Wax (Carnauba)": [
        (1063, 8,  0.40),   # C-C stretching
        (1130, 8,  0.60),   # C-C stretching
        (1296, 6,  0.35),   # CH₂ twisting
        (1440, 8,  0.40),   # CH₂ scissoring
        (1730, 8,  0.35),   # C=O ester carbonyl stretching — marker for ester waxes
        (2848, 10, 0.70),   # CH₂ symmetric stretching
        (2883, 10, 0.80),   # CH₂ asymmetric stretching
    ],
}

# ── Noise reduction method names shown in the UI combobox ─────────────────────
NOISE_METHODS = [
    "Savitzky-Golay",   # Polynomial least-squares fit in sliding window (scipy)
    "Gaussian",         # Gaussian convolution smoother (scipy or numpy fallback)
    "Moving Average",   # Uniform-kernel convolution (numpy only)
    "Median",           # Robust spike-removal filter (scipy or numpy fallback)
]


def _lorentzian(x, center, width, amp):
    """Return a Lorentzian (Cauchy) peak evaluated at array *x*.

    The Lorentzian lineshape is the natural profile of a homogeneously-broadened
    Raman peak:  I(x) = amp / (1 + ((x - center) / width)²)

    Parameters
    ----------
    x      : ndarray  — wavenumber positions to evaluate (cm⁻¹)
    center : float    — peak position (cm⁻¹)
    width  : float    — half-width at half-maximum, HWHM (cm⁻¹)
    amp    : float    — peak amplitude (a.u.)

    Returns
    -------
    ndarray — intensity at each wavenumber in *x*
    """
    return amp / (1.0 + ((x - center) / width) ** 2)


def _generate_raman_spectra(material, n_samples=20, noise=0.018, rng=None):
    """Generate *n_samples* realistic synthetic Raman spectra for *material*.

    Each spectrum is built by:
      1. Summing Lorentzian peaks from RAMAN_MATERIALS[material], with small
         random perturbations to amplitude (±10 %) and peak centre (±1.5 cm⁻¹).
      2. Adding a slight linear fluorescence background (random slope 0–4 %).
      3. Adding Gaussian white noise with standard deviation *noise*.
      4. Clipping to zero so no unphysical negative intensities appear.

    Parameters
    ----------
    material  : str   — key in RAMAN_MATERIALS
    n_samples : int   — number of spectra to generate
    noise     : float — Gaussian noise standard deviation (a.u.)
    rng       : numpy Generator or None — pass a seeded generator for reproducibility

    Returns
    -------
    ndarray, shape (n_samples, len(RAMAN_WN))
    """
    if rng is None:
        rng = np.random.default_rng(42)              # Default seed for reproducibility
    peaks = RAMAN_MATERIALS[material]                # Retrieve peak table for this material
    spectra = []
    for _ in range(n_samples):
        s = np.zeros(len(RAMAN_WN))                  # Start with a flat zero baseline
        for center, width, amp in peaks:
            # Jitter amplitude and centre position independently per spectrum
            jittered_amp    = amp    * rng.uniform(0.90, 1.10)
            jittered_center = center + rng.uniform(-1.5, 1.5)
            s += _lorentzian(RAMAN_WN, jittered_center, width, jittered_amp)
        # Linear fluorescent background: random slope, increases toward red end
        bg_slope = rng.uniform(0, 0.04)
        bg = bg_slope * (RAMAN_WN - RAMAN_WN[0]) / (RAMAN_WN[-1] - RAMAN_WN[0])
        s += bg
        s += rng.normal(0, noise, len(RAMAN_WN))     # Add Gaussian white noise
        spectra.append(np.clip(s, 0, None))           # Clip to non-negative values
    return np.array(spectra)                          # Return (n_samples, 650) array


# ═══════════════════════════════════════════════════════════════════════════════
# Main application class
# ═══════════════════════════════════════════════════════════════════════════════

class PCAApp(tk.Tk):
    """Root Tk window that owns all GUI state and analysis logic.

    The window is split into two resizable panes:
      • Left  : scrollable control panel (data source, PCA params, Raman section).
      • Right : tabbed results area (Raman viewer, scatter, variance, decision
                boundary, heatmap, data preview).

    PCA is executed on a daemon thread so the UI stays responsive during
    computation.  Results are marshalled back to the main thread via
    ``self.after(0, callback)``.

    Instance attributes (data)
    --------------------------
    data_X          : pd.DataFrame or None  — feature matrix for PCA
    data_y          : ndarray or None       — integer class labels
    feature_names   : list[str] or None     — column names matching data_X
    target_names    : list[str] or None     — human-readable class names
    pca_result      : dict or None          — last PCA run outputs (see _do_analysis)

    Instance attributes (Raman)
    ---------------------------
    raman_wavenumbers : ndarray or None   — 1-D wavenumber axis (cm⁻¹)
    raman_spectra     : ndarray or None   — 2-D intensity matrix (n, wn)
    raman_labels      : list[str] or None — per-spectrum material labels
    raman_ax          : Axes or None      — current Raman plot axes (for hover)
    _raman_vline      : Line2D or None    — vertical crosshair line object
    """

    def __init__(self):
        """Initialise the Tk root window, state variables, styles, and layout."""
        super().__init__()                      # Initialise Tk root window
        self.title("PCA Analysis Tool")         # Window title bar text
        self.geometry("1280x820")               # Default size in pixels (width x height)
        self.configure(bg=BG)                   # Apply dark background to root
        self.minsize(1100, 700)                 # Prevent layout collapse on resize

        # ── General analysis state ────────────────────────────────────────────
        self.data_X        = None   # Feature DataFrame — rows=samples, cols=features
        self.data_y        = None   # Integer label array, same length as data_X rows
        self.feature_names = None   # Column names shown in heatmap and preview
        self.target_names  = None   # Decoded class names shown in legends
        self.pca_result    = None   # Dict populated after a successful PCA run

        # ── Raman viewer state ────────────────────────────────────────────────
        self.raman_wavenumbers = None   # Shape (n_wn,) — shared x-axis for all spectra
        self.raman_spectra     = None   # Shape (n_spectra, n_wn) — raw intensity matrix
        self.raman_processed   = None   # Shape (n_spectra, n_wn) after noise/coating; None=raw
        self.raman_labels      = None   # Length n_spectra — material name per row
        self.raman_ax          = None   # Active Axes object; None before first plot
        self._raman_vline      = None   # Dashed vertical line that follows the cursor
        self._coating_vars     = {}     # Populated in _build_controls: name → BooleanVar

        self._build_styles()   # Define all ttk style overrides for the dark theme
        self._build_layout()   # Construct and place all widgets

    # ══════════════════════════════════════════════════════════════════════════
    # Styling
    # ══════════════════════════════════════════════════════════════════════════

    def _build_styles(self):
        """Configure all ttk Style rules to implement the dark theme.

        Uses the 'clam' base theme (available on all platforms) and overrides
        colours and fonts for every widget class used in the application.
        Named styles (e.g. 'Accent.TButton') can be referenced by name in
        widget constructors.
        """
        style = ttk.Style(self)
        style.theme_use("clam")   # 'clam' exposes the most colour-override options

        # ── Global defaults applied to all widgets ────────────────────────────
        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))

        # ── Frame variants ────────────────────────────────────────────────────
        style.configure("TFrame",      background=BG)       # Standard transparent frame
        style.configure("Card.TFrame", background=BG_CARD)  # Elevated card surface

        # ── Label variants ────────────────────────────────────────────────────
        style.configure("TLabel",      background=BG, foreground=TEXT,    font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"),   foreground=ACCENT)
        style.configure("Dim.TLabel",  foreground=TEXT_DIM,               font=("Segoe UI", 9))
        style.configure("Stat.TLabel", font=("Consolas", 11),             foreground=CYAN)

        # ── Primary accent button (Run PCA) ───────────────────────────────────
        style.configure(
            "Accent.TButton",
            background=ACCENT,                  # Blue fill when idle
            foreground="#000",                  # Black text for contrast on blue
            font=("Segoe UI", 10, "bold"),
            padding=(16, 8),                    # Extra horizontal padding
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_HOVER), ("disabled", BORDER)],   # State-dependent fill
            foreground=[("disabled", TEXT_DIM)],                            # Grey text when disabled
        )

        # ── Secondary button (Load, Export) ───────────────────────────────────
        style.configure(
            "TButton",
            background=BG_SECONDARY,
            foreground=TEXT,
            font=("Segoe UI", 10),
            padding=(12, 6),
            borderwidth=1,
        )
        style.map("TButton", background=[("active", BG_CARD)])   # Subtle hover lightening

        # ── Combobox ──────────────────────────────────────────────────────────
        style.configure(
            "TCombobox",
            fieldbackground=BG_SECONDARY,   # Text-field area background
            background=BG_CARD,             # Dropdown arrow button background
            foreground=TEXT,
            selectbackground=ACCENT,        # Highlighted selection fill
            selectforeground="#000",
        )
        style.map("TCombobox", fieldbackground=[("readonly", BG_SECONDARY)])

        # ── Spinbox ───────────────────────────────────────────────────────────
        style.configure(
            "TSpinbox",
            fieldbackground=BG_SECONDARY,
            background=BG_CARD,
            foreground=TEXT,
            arrowcolor=TEXT,
        )

        # ── Notebook (tabbed pane) ────────────────────────────────────────────
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=BG_SECONDARY,    # Inactive tab background
            foreground=TEXT_DIM,        # Inactive tab text
            padding=(14, 6),
            font=("Segoe UI", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", BG_CARD)],     # Active tab brightens
            foreground=[("selected", ACCENT)],       # Active tab text becomes accent blue
        )

        # ── Progress bar ──────────────────────────────────────────────────────
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG_SECONDARY)

    # ══════════════════════════════════════════════════════════════════════════
    # Layout construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        """Build the top-level window structure: title bar, separator, paned area."""
        # ── Title bar ─────────────────────────────────────────────────────────
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", padx=16, pady=(12, 4))   # Horizontal strip at top
        ttk.Label(
            title_frame, text="◆  PCA Analysis Tool",
            font=("Segoe UI", 16, "bold"), foreground=ACCENT,
        ).pack(side="left")
        ttk.Label(
            title_frame,
            text="Principal Component Analysis with Python",
            style="Dim.TLabel",
        ).pack(side="left", padx=(12, 0))   # Subtitle placed to the right of the title

        # ── Horizontal separator line ─────────────────────────────────────────
        sep = tk.Frame(self, height=1, bg=BORDER)   # 1-pixel-tall rule
        sep.pack(fill="x", padx=16, pady=(4, 8))

        # ── Resizable left/right pane ─────────────────────────────────────────
        main = ttk.PanedWindow(self, orient="horizontal")   # Draggable sash between panels
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(main, width=340)   # Control panel; fixed initial width
        main.add(left, weight=0)            # weight=0 means left pane does not expand
        self._build_controls(left)          # Populate left panel with controls

        right = ttk.Frame(main)             # Results area; expands to fill available space
        main.add(right, weight=1)           # weight=1 means right pane absorbs extra width
        self._build_results(right)          # Populate right panel with tabs and plots

    def _build_controls(self, parent):
        """Build the scrollable left panel containing all input controls.

        The panel uses a Canvas + Scrollbar pattern to allow the control list to
        grow beyond the panel height without clipping.  All actual control widgets
        are parented to *scroll_frame* (which is embedded in the canvas).

        Parameters
        ----------
        parent : ttk.Frame — the left pane of the main PanedWindow
        """
        # ── Scrollable canvas + scrollbar setup ───────────────────────────────
        canvas    = tk.Canvas(parent, bg=BG, highlightthickness=0, width=320)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)   # All controls live inside this frame

        # Resize the canvas scroll-region when the frame's content changes height
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        # Embed scroll_frame at the top-left corner of the canvas
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=310)
        canvas.configure(yscrollcommand=scrollbar.set)   # Link scrollbar to canvas

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Translate mouse-wheel delta (120 units per notch on Windows) into scroll units
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Global binding (no focus needed)

        p = scroll_frame   # Shorthand alias used throughout this method

        # ── Section: Data Source ──────────────────────────────────────────────
        self._section(p, "DATA SOURCE")

        ttk.Label(p, text="Preset Dataset").pack(anchor="w", padx=8, pady=(4, 2))
        self.preset_var = tk.StringVar(value="-- Select Preset --")   # Tracks combobox selection
        presets = ttk.Combobox(
            p, textvariable=self.preset_var, state="readonly", width=34,
            values=[
                "-- Select Preset --",
                "Iris (4 features, 3 classes)",
                "Wine (13 features, 3 classes)",
                "Breast Cancer (30 features, 2 classes)",
                "Random Blobs (5 features, 4 classes)",
                "Random Circles (2 features, 2 classes)",
            ],
        )
        presets.pack(padx=8, pady=2, fill="x")
        presets.bind("<<ComboboxSelected>>", self._on_preset_selected)   # Fire on selection change

        ttk.Label(p, text="─ or ─", foreground=TEXT_DIM, font=("Segoe UI", 9)).pack(pady=4)
        ttk.Button(p, text="📂  Load CSV File", command=self._load_csv).pack(padx=8, fill="x")

        # Dynamic label that updates after data is loaded
        self.data_info_var = tk.StringVar(value="No data loaded")
        ttk.Label(
            p, textvariable=self.data_info_var,
            style="Dim.TLabel", wraplength=280,
        ).pack(padx=8, pady=(6, 2), anchor="w")

        # ── Section: PCA Parameters ───────────────────────────────────────────
        self._section(p, "PCA PARAMETERS")

        ttk.Label(p, text="Number of Components").pack(anchor="w", padx=8, pady=(4, 2))
        self.n_components_var = tk.IntVar(value=2)   # Default to 2-D projection
        spin = ttk.Spinbox(p, from_=1, to=50, textvariable=self.n_components_var, width=8)
        spin.pack(anchor="w", padx=8)

        ttk.Label(p, text="Test Split Ratio").pack(anchor="w", padx=8, pady=(8, 2))
        self.test_ratio_var = tk.DoubleVar(value=0.2)   # 20 % held-out test set by default
        ratio_frame = ttk.Frame(p)
        ratio_frame.pack(fill="x", padx=8)
        self.ratio_scale = tk.Scale(
            ratio_frame, from_=0.05, to=0.5, resolution=0.05,
            orient="horizontal", variable=self.test_ratio_var,
            bg=BG, fg=TEXT, troughcolor=BG_SECONDARY,
            highlightthickness=0, sliderrelief="flat",
            activebackground=ACCENT, font=("Consolas", 9),
        )
        self.ratio_scale.pack(fill="x")

        # Standardisation is strongly recommended for PCA; enabled by default
        self.standardize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            p, text="Standardize features (recommended)",
            variable=self.standardize_var,
        ).pack(anchor="w", padx=8, pady=(6, 0))

        # Logistic Regression on the PCA-projected space; optional
        self.classify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            p, text="Run Logistic Regression classifier",
            variable=self.classify_var,
        ).pack(anchor="w", padx=8, pady=(2, 0))

        # ── Section: Raman Data ───────────────────────────────────────────────
        self._section(p, "RAMAN DATA")

        ttk.Label(p, text="Raman Preset").pack(anchor="w", padx=8, pady=(4, 2))
        self.raman_preset_var = tk.StringVar(value="-- Select Raman Preset --")
        raman_combo = ttk.Combobox(
            p, textvariable=self.raman_preset_var, state="readonly", width=34,
            values=["-- Select Raman Preset --"] + list(RAMAN_PRESETS.keys()),
        )
        raman_combo.pack(padx=8, pady=2, fill="x")
        raman_combo.bind("<<ComboboxSelected>>", self._on_raman_preset_selected)

        ttk.Label(p, text="─ or ─", foreground=TEXT_DIM, font=("Segoe UI", 9)).pack(pady=4)
        ttk.Button(p, text="📂  Load Raman CSV", command=self._load_raman_csv).pack(padx=8, fill="x")

        # Normalise: rescale each spectrum so its maximum intensity equals 1
        self.raman_normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            p, text="Normalize spectra (max = 1)",
            variable=self.raman_normalize_var,
            command=self._plot_raman_spectra,   # Re-draw immediately on toggle
        ).pack(anchor="w", padx=8, pady=(8, 0))

        # Stack: add a vertical offset proportional to class index so overlapping
        # spectra can be distinguished without normalisation
        self.raman_stack_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            p, text="Stack with vertical offset",
            variable=self.raman_stack_var,
            command=self._plot_raman_spectra,   # Re-draw immediately on toggle
        ).pack(anchor="w", padx=8, pady=(2, 4))

        # ── Section: Signal Processing ────────────────────────────────────────
        self._section(p, "SIGNAL PROCESSING")

        # ─ Noise reduction ─
        ttk.Label(p, text="Noise Reduction", font=("Segoe UI", 9, "bold"),
                  foreground=TEXT_DIM).pack(anchor="w", padx=8, pady=(6, 2))

        ttk.Label(p, text="Method").pack(anchor="w", padx=8)
        self.noise_method_var = tk.StringVar(value="Savitzky-Golay")
        noise_combo = ttk.Combobox(
            p, textvariable=self.noise_method_var, state="readonly", width=24,
            values=NOISE_METHODS,
        )
        noise_combo.pack(padx=8, pady=2, fill="x")

        # Window/kernel size — must be odd for SG and median filters
        win_row = ttk.Frame(p)
        win_row.pack(fill="x", padx=8, pady=2)
        ttk.Label(win_row, text="Window size:").pack(side="left")
        self.noise_window_var = tk.IntVar(value=11)   # 11-point default is gentle
        ttk.Spinbox(
            win_row, from_=3, to=101, increment=2,    # Increment by 2 to keep odd values
            textvariable=self.noise_window_var, width=6,
        ).pack(side="left", padx=(6, 0))

        nr_row = ttk.Frame(p)
        nr_row.pack(fill="x", padx=8, pady=(2, 4))
        ttk.Button(nr_row, text="Apply Noise Reduction",
                   command=self._apply_noise_reduction).pack(side="left", fill="x", expand=True)

        # ─ Coating simulation ─
        ttk.Label(p, text="Coating Simulation", font=("Segoe UI", 9, "bold"),
                  foreground=TEXT_DIM).pack(anchor="w", padx=8, pady=(8, 2))

        # One checkbox per coating; stored in self._coating_vars for iteration
        for coat_name in COATING_CATALOGUE:
            var = tk.BooleanVar(value=False)
            self._coating_vars[coat_name] = var
            ttk.Checkbutton(p, text=coat_name, variable=var).pack(anchor="w", padx=12)

        # Thickness slider — scales the coating peak amplitude (0 = off, 1 = full)
        ttk.Label(p, text="Coating thickness (0–1)").pack(anchor="w", padx=8, pady=(8, 0))
        self.coat_thickness_var = tk.DoubleVar(value=0.3)
        tk.Scale(
            p, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
            variable=self.coat_thickness_var,
            bg=BG, fg=TEXT, troughcolor=BG_SECONDARY,
            highlightthickness=0, sliderrelief="flat",
            activebackground=ACCENT, font=("Consolas", 9),
        ).pack(fill="x", padx=8)

        # Attenuation slider — fraction of substrate signal blocked by the coating
        ttk.Label(p, text="Substrate attenuation (0–0.9)").pack(anchor="w", padx=8, pady=(6, 0))
        self.coat_attenuation_var = tk.DoubleVar(value=0.2)
        tk.Scale(
            p, from_=0.0, to=0.9, resolution=0.05, orient="horizontal",
            variable=self.coat_attenuation_var,
            bg=BG, fg=TEXT, troughcolor=BG_SECONDARY,
            highlightthickness=0, sliderrelief="flat",
            activebackground=ACCENT, font=("Consolas", 9),
        ).pack(fill="x", padx=8)

        coat_row = ttk.Frame(p)
        coat_row.pack(fill="x", padx=8, pady=(4, 2))
        ttk.Button(coat_row, text="Apply Coating",
                   command=self._apply_coating_simulation).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(coat_row, text="Reset",
                   command=self._reset_processing).pack(side="left", fill="x", expand=True)

        # ── Section: Run button + progress bar ────────────────────────────────
        self._section(p, "")   # Empty title = just a horizontal rule
        self.run_btn = ttk.Button(
            p, text="▶  Run PCA Analysis",
            style="Accent.TButton", command=self._run_analysis,
        )
        self.run_btn.pack(padx=8, pady=4, fill="x")

        # Indeterminate progress bar (spinning) shown while PCA runs on thread
        self.progress = ttk.Progressbar(p, mode="indeterminate", style="Horizontal.TProgressbar")
        self.progress.pack(padx=8, fill="x", pady=(0, 4))

        # ── Section: Results summary text box ─────────────────────────────────
        self._section(p, "RESULTS SUMMARY")
        self.stats_text = tk.Text(
            p, height=14,
            bg=BG_SECONDARY, fg=CYAN,            # Monospace cyan text on dark background
            font=("Consolas", 9), relief="flat", bd=0,
            insertbackground=CYAN,               # Cursor colour
            selectbackground=ACCENT,
            wrap="word", padx=8, pady=6,
        )
        self.stats_text.pack(padx=8, fill="x", pady=(2, 8))
        self.stats_text.insert("1.0", "Run an analysis to see results...")
        self.stats_text.config(state="disabled")   # Read-only until updated by analysis

    # ══════════════════════════════════════════════════════════════════════════
    # Results area (right panel)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_results(self, parent):
        """Build the notebook (tabbed pane) that holds all result visualisations.

        Tabs created:
          1. Raman Spectra   — interactive spectrum viewer
          2. PCA Scatter     — 2-D (or 1-D) scatter of projected training data
          3. Variance        — scree plot + cumulative variance
          4. Decision Boundary — filled contour + scatter for train and test
          5. Component Heatmap — colour-coded loading matrix
          6. Data Preview    — first 20 rows + descriptive statistics

        Parameters
        ----------
        parent : ttk.Frame — right pane of the main PanedWindow
        """
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        # ── Create all tab frames ─────────────────────────────────────────────
        self.tab_raman    = ttk.Frame(self.notebook)   # Raman spectrum viewer
        self.tab_scatter  = ttk.Frame(self.notebook)   # PCA scatter plot
        self.tab_variance = ttk.Frame(self.notebook)   # Scree + cumulative variance
        self.tab_decision = ttk.Frame(self.notebook)   # Decision boundary (train / test)
        self.tab_heatmap  = ttk.Frame(self.notebook)   # Component loading heatmap
        self.tab_data     = ttk.Frame(self.notebook)   # Raw data preview table

        # Register tabs in display order
        self.notebook.add(self.tab_raman,    text="  Raman Spectra  ")
        self.notebook.add(self.tab_scatter,  text="  PCA Scatter  ")
        self.notebook.add(self.tab_variance, text="  Variance  ")
        self.notebook.add(self.tab_decision, text="  Decision Boundary  ")
        self.notebook.add(self.tab_heatmap,  text="  Component Heatmap  ")
        self.notebook.add(self.tab_data,     text="  Data Preview  ")

        self._build_raman_tab(self.tab_raman)   # Raman tab has special canvas + hover

        # ── Create matplotlib figures for the four PCA plot tabs ──────────────
        self.figures  = {}   # name → Figure object
        self.canvases = {}   # name → FigureCanvasTkAgg object

        for name, tab in [
            ("scatter",  self.tab_scatter),
            ("variance", self.tab_variance),
            ("decision", self.tab_decision),
            ("heatmap",  self.tab_heatmap),
        ]:
            fig = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
            canvas = FigureCanvasTkAgg(fig, master=tab)

            # Build navigation toolbar and apply dark colours to all children
            toolbar = NavigationToolbar2Tk(canvas, tab)
            toolbar.config(background=BG_SECONDARY)
            for child in toolbar.winfo_children():
                try:
                    child.config(background=BG_SECONDARY)   # Colour toolbar buttons
                except Exception:
                    pass                                     # Separators have no bg option
            toolbar.update()

            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name]  = fig
            self.canvases[name] = canvas

        # ── Data preview tab — scrolled text widget ───────────────────────────
        self.data_text = scrolledtext.ScrolledText(
            self.tab_data,
            bg=BG_SECONDARY, fg=TEXT,
            font=("Consolas", 9), relief="flat", bd=0,
            insertbackground=TEXT, selectbackground=ACCENT,
        )
        self.data_text.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Draw placeholder text on each PCA plot until analysis is run ──────
        for name, fig in self.figures.items():
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)
            ax.text(
                0.5, 0.5, "Load data and run analysis",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color=TEXT_DIM, style="italic",
            )
            ax.set_xticks([])                          # Hide axis tick marks
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)               # Remove box border
            self.canvases[name].draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Shared UI helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _section(self, parent, title):
        """Render a labelled horizontal rule as a visual section divider.

        If *title* is an empty string only the rule is drawn, providing a
        blank spacer between the parameter section and the Run button.

        Parameters
        ----------
        parent : tk widget — container in which to pack the divider
        title  : str       — uppercase section label; empty string → rule only
        """
        if title:
            f = ttk.Frame(parent)
            f.pack(fill="x", padx=8, pady=(12, 2))
            ttk.Label(
                f, text=title,
                font=("Segoe UI", 9, "bold"), foreground=TEXT_DIM,
            ).pack(side="left")
            sep = tk.Frame(f, height=1, bg=BORDER)   # 1-px horizontal rule
            sep.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=1)

    def _style_ax(self, ax, title=""):
        """Apply the dark theme to a matplotlib Axes object.

        Sets the axes background, spine colour, tick colour, and title.  Call
        this immediately after ``fig.add_subplot()`` before drawing data.

        Parameters
        ----------
        ax    : matplotlib Axes — target axes to style
        title : str             — axes title; omit to leave blank
        """
        ax.set_facecolor(BG_CARD)                        # Axes interior background
        ax.tick_params(colors=TEXT_DIM, labelsize=8)     # Tick marks and labels
        for spine in ax.spines.values():
            spine.set_color(BORDER)                      # Subtle border around plot area
        if title:
            ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.xaxis.label.set_color(TEXT_DIM)               # Axis label colour
        ax.yaxis.label.set_color(TEXT_DIM)

    def _update_stats(self, text):
        """Replace all content in the results summary text box.

        The widget is kept read-only (state='disabled') at all times except
        during this update to prevent accidental user edits.

        Parameters
        ----------
        text : str — multi-line string to display
        """
        self.stats_text.config(state="normal")      # Temporarily unlock for writing
        self.stats_text.delete("1.0", "end")        # Clear existing content
        self.stats_text.insert("1.0", text)         # Insert new content at start
        self.stats_text.config(state="disabled")    # Re-lock to read-only

    # ══════════════════════════════════════════════════════════════════════════
    # Data loading — general
    # ══════════════════════════════════════════════════════════════════════════

    def _on_preset_selected(self, event=None):
        """Load one of the bundled sklearn datasets when the combobox fires.

        After loading, updates *data_X*, *data_y*, *feature_names*,
        *target_names*, sets a sensible default component count, and refreshes
        the data preview tab.

        Parameters
        ----------
        event : tk.Event or None — combobox '<<ComboboxSelected>>' event (unused)
        """
        sel = self.preset_var.get()   # Read the currently selected display string

        if "Iris" in sel:
            d = load_iris()                                          # 150 samples, 4 features, 3 classes
            self.data_X        = pd.DataFrame(d.data, columns=d.feature_names)
            self.data_y        = d.target
            self.feature_names = list(d.feature_names)
            self.target_names  = list(d.target_names)

        elif "Wine" in sel:
            d = load_wine()                                          # 178 samples, 13 features, 3 classes
            self.data_X        = pd.DataFrame(d.data, columns=d.feature_names)
            self.data_y        = d.target
            self.feature_names = list(d.feature_names)
            self.target_names  = list(d.target_names)

        elif "Breast Cancer" in sel:
            d = load_breast_cancer()                                 # 569 samples, 30 features, 2 classes
            self.data_X        = pd.DataFrame(d.data, columns=d.feature_names)
            self.data_y        = d.target
            self.feature_names = list(d.feature_names)
            self.target_names  = list(d.target_names)

        elif "Blobs" in sel:
            from sklearn.datasets import make_blobs
            X, y = make_blobs(n_samples=400, n_features=5, centers=4, random_state=42)
            self.feature_names = [f"Feature_{i+1}" for i in range(5)]
            self.data_X        = pd.DataFrame(X, columns=self.feature_names)
            self.data_y        = y
            self.target_names  = [f"Cluster {i}" for i in range(4)]

        elif "Circles" in sel:
            from sklearn.datasets import make_circles
            # factor controls radius ratio of inner/outer circle
            X, y = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=42)
            self.feature_names = ["X1", "X2"]
            self.data_X        = pd.DataFrame(X, columns=self.feature_names)
            self.data_y        = y
            self.target_names  = ["Inner", "Outer"]

        else:
            return   # Placeholder row selected — do nothing

        n_samples, n_features = self.data_X.shape
        n_classes = len(np.unique(self.data_y))
        # Update the info label shown below the combobox
        self.data_info_var.set(
            f"✓ {sel.split('(')[0].strip()}: {n_samples} samples, "
            f"{n_features} features, {n_classes} classes"
        )
        self.n_components_var.set(min(2, n_features))   # Cap default at 2 (or fewer if needed)
        self._show_data_preview()

    def _load_csv(self):
        """Open a file-picker dialog and load a general-purpose CSV dataset.

        Convention: last column = class label; all preceding columns = features.
        String labels are integer-encoded via LabelEncoder.  Shows an error
        dialog if the file cannot be parsed.
        """
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return   # User cancelled — do nothing

        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                messagebox.showerror("Error", "CSV must have at least 2 columns.")
                return

            # Split into features (all but last) and target (last)
            self.feature_names = list(df.columns[:-1])
            self.data_X        = df.iloc[:, :-1]
            self.data_y        = df.iloc[:, -1].values

            # Encode string labels to contiguous integers
            if self.data_y.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.target_names = list(np.unique(self.data_y))     # Original string names
                self.data_y       = le.fit_transform(self.data_y)    # Replace with 0, 1, 2, …
            else:
                self.target_names = [str(c) for c in np.unique(self.data_y)]

            n_samples, n_features = self.data_X.shape
            n_classes = len(np.unique(self.data_y))
            name = path.split("/")[-1].split("\\")[-1]   # Extract bare filename (cross-platform)
            self.data_info_var.set(
                f"✓ {name}: {n_samples} samples, "
                f"{n_features} features, {n_classes} classes"
            )
            self.n_components_var.set(min(2, n_features))
            self._show_data_preview()
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))

    def _show_data_preview(self):
        """Populate the 'Data Preview' tab with shape, first 20 rows, and statistics.

        Uses an in-memory StringIO buffer to format the DataFrame text cheaply
        without creating temporary files.
        """
        self.data_text.delete("1.0", "end")   # Clear stale content
        if self.data_X is None:
            return   # Nothing to show yet

        buf = io.StringIO()                    # In-memory text buffer
        df  = self.data_X.copy()
        df["target"] = self.data_y             # Append label column for context

        buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n")
        buf.write("=" * 60 + "\n\n")
        buf.write("First 20 rows:\n")
        buf.write(df.head(20).to_string(index=True))   # Pandas tabular string
        buf.write("\n\n" + "=" * 60 + "\n")
        buf.write("\nDescriptive Statistics:\n")
        buf.write(df.describe().to_string())            # Count, mean, std, percentiles

        self.data_text.insert("1.0", buf.getvalue())   # Insert formatted text at top

    # ══════════════════════════════════════════════════════════════════════════
    # PCA analysis
    # ══════════════════════════════════════════════════════════════════════════

    def _run_analysis(self):
        """Validate data presence then launch PCA on a background thread.

        The Run button is disabled and the progress bar is started here so the
        user gets immediate feedback.  The thread calls _do_analysis(); on
        completion or error it re-enables the button via self.after().
        """
        if self.data_X is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return

        self.run_btn.config(state="disabled")    # Prevent concurrent runs
        self.progress.start(10)                  # Animate progress bar (10 ms interval)

        thread = threading.Thread(target=self._do_analysis, daemon=True)
        thread.start()   # daemon=True: thread dies if main window is closed

    def _do_analysis(self):
        """Execute the full PCA (and optional classification) pipeline.

        Runs on a background thread.  All widget updates are marshalled back to
        the main thread using ``self.after(0, callback)`` to avoid Tk
        thread-safety issues.

        Pipeline
        ────────
        1. Optional StandardScaler — zero-mean, unit-variance scaling.
        2. Train/test split with optional stratification.
        3. PCA on the training set; transform both splits.
        4. Second full PCA (all components) on the whole dataset for the scree plot.
        5. Optional LogisticRegression on PCA-projected training data.
        6. Build results dict and schedule UI updates on the main thread.
        """
        try:
            # ── Read current parameter values from the UI ─────────────────────
            X              = self.data_X.values.astype(float)   # Convert DataFrame to float ndarray
            y              = self.data_y
            n_comp         = self.n_components_var.get()
            test_ratio     = self.test_ratio_var.get()
            do_standardize = self.standardize_var.get()
            do_classify    = self.classify_var.get()

            # Cap component count at the minimum of (n_features, n_samples)
            n_comp = min(n_comp, X.shape[1], X.shape[0])

            # ── Train / test split ────────────────────────────────────────────
            # stratify=y ensures proportional class representation in both splits
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_ratio,
                random_state=0,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            # ── Optional standardisation ──────────────────────────────────────
            # Fit scaler on training data only; apply the same transform to test
            if do_standardize:
                scaler      = StandardScaler()
                X_train_s   = scaler.fit_transform(X_train)   # Fit + transform train
                X_test_s    = scaler.transform(X_test)        # Only transform test
            else:
                X_train_s   = X_train   # Pass through unchanged
                X_test_s    = X_test

            # ── PCA ───────────────────────────────────────────────────────────
            pca          = PCA(n_components=n_comp)
            X_train_pca  = pca.fit_transform(X_train_s)   # Fit on train, project train
            X_test_pca   = pca.transform(X_test_s)        # Project test using train's axes

            explained    = pca.explained_variance_ratio_   # Fraction of variance per component
            cumulative   = np.cumsum(explained)             # Running total for cumulative plot

            # Full PCA (all components) to build a complete scree plot
            max_comp     = min(X.shape[1], X.shape[0])
            pca_full     = PCA(n_components=max_comp)
            if do_standardize:
                pca_full.fit(scaler.fit_transform(X))   # Re-fit scaler on full dataset
            else:
                pca_full.fit(X)
            full_explained = pca_full.explained_variance_ratio_

            # ── Build statistics summary string ───────────────────────────────
            stats_lines = []
            stats_lines.append(f"Components: {n_comp}")
            stats_lines.append(f"Train/Test: {len(y_train)}/{len(y_test)}")
            stats_lines.append("─────────────────────────")
            for i, v in enumerate(explained):
                stats_lines.append(f"PC{i+1} variance: {v:.4f} ({v*100:.1f}%)")
            stats_lines.append("─────────────────────────")
            stats_lines.append(f"Total explained: {cumulative[-1]*100:.1f}%")

            # ── Optional classifier ───────────────────────────────────────────
            classifier = None
            y_pred     = None
            accuracy   = None

            if do_classify and len(np.unique(y)) > 1:
                classifier = LogisticRegression(random_state=0, max_iter=1000)
                classifier.fit(X_train_pca, y_train)           # Train on projected data
                y_pred     = classifier.predict(X_test_pca)    # Predict on projected test
                accuracy   = accuracy_score(y_test, y_pred)

                cm = confusion_matrix(y_test, y_pred)
                stats_lines.append(f"\nClassifier Accuracy: {accuracy*100:.1f}%")
                stats_lines.append("\nConfusion Matrix:")
                stats_lines.append(str(cm))

            # ── Store results for plotting ─────────────────────────────────────
            self.pca_result = {
                "pca":          pca,            # Fitted PCA object (has .components_)
                "X_train_pca":  X_train_pca,    # Projected training data
                "X_test_pca":   X_test_pca,     # Projected test data
                "y_train":      y_train,
                "y_test":       y_test,
                "explained":    explained,       # Per-component explained variance ratios
                "cumulative":   cumulative,      # Cumulative explained variance
                "full_explained": full_explained,# All-component ratios for scree plot
                "classifier":   classifier,      # LogisticRegression or None
                "y_pred":       y_pred,
                "accuracy":     accuracy,
            }

            # ── Schedule UI updates on the main thread ────────────────────────
            self.after(0, lambda: self._update_stats("\n".join(stats_lines)))
            self.after(0, self._plot_all)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
        finally:
            # Always re-enable UI regardless of success or failure
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.run_btn.config(state="normal"))

    # ══════════════════════════════════════════════════════════════════════════
    # PCA result plots
    # ══════════════════════════════════════════════════════════════════════════

    def _plot_all(self):
        """Trigger all four PCA result plots in sequence.

        Called from the main thread via self.after() once _do_analysis completes.
        """
        self._plot_scatter()    # Scatter of projected data coloured by class
        self._plot_variance()   # Scree + cumulative variance
        self._plot_decision()   # Decision boundary (train / test panels)
        self._plot_heatmap()    # Component loading matrix

    def _plot_scatter(self):
        """Draw the PCA scatter plot on the 'PCA Scatter' tab.

        Shows the first two principal components (or first one if n_comp == 1)
        with each class in a distinct colour.  Only training data is plotted.
        """
        r   = self.pca_result
        fig = self.figures["scatter"]
        fig.clf()   # Clear previous drawing

        X_pca   = r["X_train_pca"]   # Shape (n_train, n_comp)
        y       = r["y_train"]
        classes = np.unique(y)

        if X_pca.shape[1] >= 2:
            # ── Two-dimensional scatter (PC1 vs PC2) ─────────────────────────
            ax = fig.add_subplot(111)
            self._style_ax(ax, "PCA — First Two Components")
            for i, cls in enumerate(classes):
                mask  = y == cls                                           # Boolean index for this class
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = self.target_names[cls] if cls < len(self.target_names) else f"Class {cls}"
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=label,
                    alpha=0.75, s=40,
                    edgecolors="white", linewidths=0.3,   # White ring improves visibility
                )
            ax.set_xlabel("PC 1", fontsize=10)
            ax.set_ylabel("PC 2", fontsize=10)
            ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER, fontsize=9, labelcolor=TEXT)
        else:
            # ── One-dimensional scatter (PC1 only, y=0) ──────────────────────
            ax = fig.add_subplot(111)
            self._style_ax(ax, "PCA — First Component")
            for i, cls in enumerate(classes):
                mask  = y == cls
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = self.target_names[cls] if cls < len(self.target_names) else f"Class {cls}"
                ax.scatter(
                    X_pca[mask, 0], np.zeros(mask.sum()),   # All points at y=0
                    c=color, label=label,
                    alpha=0.75, s=40,
                    edgecolors="white", linewidths=0.3,
                )
            ax.set_xlabel("PC 1", fontsize=10)
            ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER, fontsize=9, labelcolor=TEXT)

        self.canvases["scatter"].draw()   # Flush the Tk canvas

    def _plot_variance(self):
        """Draw the scree plot and cumulative variance plot on the 'Variance' tab.

        Left panel  : bar chart of individual component variance (scree plot)
                      with a line overlay showing the "elbow".
        Right panel : cumulative sum of explained variance with a dashed
                      horizontal line at the 95 % threshold.

        Uses the full PCA (all components) computed during analysis.
        """
        r        = self.pca_result
        fig      = self.figures["variance"]
        fig.clf()

        full_exp = r["full_explained"]           # All-component explained variance ratios
        n        = len(full_exp)                 # Total number of components
        x        = np.arange(1, n + 1)           # Component index labels (1-based)

        # ── Left: scree plot ──────────────────────────────────────────────────
        ax1 = fig.add_subplot(121)
        self._style_ax(ax1, "Scree Plot")
        ax1.bar(x, full_exp * 100, color=ACCENT, alpha=0.8, edgecolor="none")
        ax1.plot(x, full_exp * 100, "o-", color=ORANGE, markersize=4, linewidth=1.5)
        ax1.set_xlabel("Component", fontsize=9)
        ax1.set_ylabel("Variance Explained (%)", fontsize=9)
        if n <= 20:
            ax1.set_xticks(x)   # Avoid crowded tick labels for high-dim data

        # ── Right: cumulative variance ────────────────────────────────────────
        ax2 = fig.add_subplot(122)
        self._style_ax(ax2, "Cumulative Variance")
        cum = np.cumsum(full_exp) * 100          # Running sum converted to percent
        ax2.fill_between(x, cum, color=GREEN, alpha=0.2)   # Shaded area under the curve
        ax2.plot(x, cum, "o-", color=GREEN, markersize=4, linewidth=2)
        ax2.axhline(y=95, color=RED, linestyle="--", linewidth=1, alpha=0.7)   # 95 % guideline
        ax2.text(n * 0.7, 96, "95% threshold", color=RED, fontsize=8)
        ax2.set_xlabel("Component", fontsize=9)
        ax2.set_ylabel("Cumulative Variance (%)", fontsize=9)
        ax2.set_ylim(0, 105)   # Leave headroom above 100 % for readability
        if n <= 20:
            ax2.set_xticks(x)

        fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96)
        self.canvases["variance"].draw()

    def _plot_decision(self):
        """Draw filled decision-boundary contours on the 'Decision Boundary' tab.

        Requires the classifier and at least 2 PCA components.  Two side-by-side
        sub-panels show the training set boundary and the test set boundary.

        The boundary is computed by re-training a fresh LogisticRegression on
        the first two PCA components (regardless of n_comp) so the 2-D
        boundary always corresponds to the displayed scatter axes.
        """
        r          = self.pca_result
        fig        = self.figures["decision"]
        fig.clf()

        classifier = r["classifier"]

        # Guard: show a message if classifier is disabled or n_comp < 2
        if classifier is None or r["X_train_pca"].shape[1] < 2:
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)
            ax.text(
                0.5, 0.5,
                "Enable classifier with ≥2 components\nto see decision boundaries",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=TEXT_DIM,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.canvases["decision"].draw()
            return

        # Use only the first two PCA dimensions for visualisation
        X_train = r["X_train_pca"][:, :2]
        y_train = r["y_train"]
        X_test  = r["X_test_pca"][:, :2]
        y_test  = r["y_test"]
        classes = np.unique(y_train)

        # Retrain a 2-D classifier specifically for boundary drawing
        clf2d = LogisticRegression(random_state=0, max_iter=1000)
        clf2d.fit(X_train, y_train)

        for idx, (X_set, y_set, title_str) in enumerate([
            (X_train, y_train, "Decision Boundary (Train)"),
            (X_test,  y_test,  "Decision Boundary (Test)"),
        ]):
            ax = fig.add_subplot(1, 2, idx + 1)
            self._style_ax(ax, title_str)

            # Build a dense mesh over the data extent to classify every pixel
            h     = 0.05                                          # Mesh cell size (PC units)
            x_min = X_set[:, 0].min() - 1
            x_max = X_set[:, 0].max() + 1
            y_min = X_set[:, 1].min() - 1
            y_max = X_set[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, h),
                np.arange(y_min, y_max, h),
            )
            Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()])   # Classify every mesh point
            Z = Z.reshape(xx.shape)                             # Reshape to 2-D grid

            cmap_bg = PLOT_CMAPS[idx % len(PLOT_CMAPS)]          # Alternate colormaps
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)     # Filled decision regions

            # Overlay the actual data points
            for i, cls in enumerate(classes):
                mask  = y_set == cls
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = self.target_names[cls] if cls < len(self.target_names) else f"Class {cls}"
                ax.scatter(
                    X_set[mask, 0], X_set[mask, 1],
                    c=color, label=label,
                    s=30, edgecolors="white", linewidths=0.3, alpha=0.85,
                )
            ax.set_xlabel("PC 1", fontsize=9)
            ax.set_ylabel("PC 2", fontsize=9)
            ax.legend(
                facecolor=BG_SECONDARY, edgecolor=BORDER,
                fontsize=8, labelcolor=TEXT, loc="best",
            )

        fig.subplots_adjust(wspace=0.3, left=0.08, right=0.96)
        self.canvases["decision"].draw()

    def _plot_heatmap(self):
        """Draw the PCA component loading matrix on the 'Component Heatmap' tab.

        Each row = one principal component.
        Each column = one original feature.
        Cell colour encodes the loading (contribution weight) — blue for negative,
        red for positive (RdBu_r colourmap).

        Cell values are annotated if the total cell count ≤ 120 (12 × 10).
        """
        r          = self.pca_result
        fig        = self.figures["heatmap"]
        fig.clf()

        pca        = r["pca"]
        components = pca.components_               # Shape (n_comp, n_features)
        n_comp     = components.shape[0]
        n_feat     = components.shape[1]

        ax = fig.add_subplot(111)
        self._style_ax(ax, "PCA Component Loadings")

        # Truncate feature labels longer than 18 chars to keep the plot legible
        feat_labels = self.feature_names[:n_feat] if self.feature_names else [f"F{i}" for i in range(n_feat)]
        feat_labels = [l[:18] + "…" if len(l) > 18 else l for l in feat_labels]

        # Draw the colour-encoded loading matrix
        im = ax.imshow(components, cmap="RdBu_r", aspect="auto", interpolation="nearest")
        ax.set_yticks(range(n_comp))
        ax.set_yticklabels([f"PC{i+1}" for i in range(n_comp)], fontsize=9, color=TEXT)
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=7, color=TEXT_DIM)

        # Colour bar showing the loading scale
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
        cbar.outline.set_edgecolor(BORDER)

        # Annotate each cell with its numeric loading value if grid is small enough
        if n_comp * n_feat <= 120:
            for i in range(n_comp):
                for j in range(n_feat):
                    val   = components[i, j]
                    color = "#000" if abs(val) > 0.4 else TEXT_DIM   # Dark text on strong cells
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

        fig.subplots_adjust(bottom=0.22, left=0.1, right=0.95)
        self.canvases["heatmap"].draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Raman viewer
    # ══════════════════════════════════════════════════════════════════════════

    def _build_raman_tab(self, parent):
        """Construct the interactive Raman spectrum viewer inside *parent*.

        Layout:
          • Header label at the top.
          • matplotlib Figure with NavigationToolbar (zoom / pan / save).
          • Single-line cursor readout label below the canvas.

        The cursor readout is updated on every mouse-move via the
        'motion_notify_event' callback.

        Parameters
        ----------
        parent : ttk.Frame — the 'Raman Spectra' notebook tab frame
        """
        # ── Header strip ──────────────────────────────────────────────────────
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(ctrl, text="Interactive Raman Viewer", style="Header.TLabel").pack(side="left")

        # ── matplotlib figure embedded in the Tk frame ────────────────────────
        self.raman_fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_CARD)
        self.raman_fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10)
        self.raman_canvas = FigureCanvasTkAgg(self.raman_fig, master=parent)

        # Apply dark colours to the navigation toolbar
        toolbar = NavigationToolbar2Tk(self.raman_canvas, parent)
        toolbar.config(background=BG_SECONDARY)
        for child in toolbar.winfo_children():
            try:
                child.config(background=BG_SECONDARY)   # Theme each toolbar button
            except Exception:
                pass                                     # Skip widgets that reject bg
        toolbar.update()

        self.raman_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Cursor readout ────────────────────────────────────────────────────
        self.raman_cursor_var = tk.StringVar(
            value="Hover over the plot — wavenumber and intensity will appear here"
        )
        ttk.Label(parent, textvariable=self.raman_cursor_var,
                  style="Dim.TLabel").pack(pady=(2, 6))

        # Register hover callback (fires on every mouse-move over the canvas)
        self.raman_canvas.mpl_connect("motion_notify_event", self._on_raman_hover)

        # ── Placeholder axes (shown before any data is loaded) ────────────────
        ax = self.raman_fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        ax.text(
            0.5, 0.5, "Select a Raman preset or load a CSV to view spectra",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color=TEXT_DIM, style="italic",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.raman_canvas.draw()

    def _on_raman_preset_selected(self, event=None):
        """Generate synthetic spectra for the chosen preset and load them.

        Reads the current combobox value, looks it up in RAMAN_PRESETS, generates
        the requested number of spectra per material using a fixed random seed,
        and passes the result to _set_raman_data().

        Parameters
        ----------
        event : tk.Event or None — combobox '<<ComboboxSelected>>' event (unused)
        """
        sel    = self.raman_preset_var.get()
        groups = RAMAN_PRESETS.get(sel)   # Returns list of (material, n) or None for placeholder
        if not groups:
            return

        rng = np.random.default_rng(0)   # Fixed seed for reproducible preset spectra
        all_spectra = []
        all_labels  = []

        for material, n in groups:
            sp = _generate_raman_spectra(material, n_samples=n, rng=rng)
            all_spectra.append(sp)                 # Accumulate (n, 650) arrays
            all_labels.extend([material] * n)      # One label string per spectrum

        self._set_raman_data(
            RAMAN_WN,
            np.vstack(all_spectra),   # Stack into (total_n, 650)
            all_labels,
        )

    def _load_raman_csv(self):
        """Open a file-picker dialog and load a Raman spectroscopy CSV file.

        Two formats are auto-detected:

        **Format A** (most common — e.g. Renishaw, WiTec, LabSpec exports)::

            Wavenumber, Sample_1, Sample_2, …
            100.0,      0.001,    0.002,    …
            …

        **Format B** (transposed — rows are spectra)::

            label,   100.0, 200.0, … , 3300.0
            Silicon, 0.001, 0.003, … , 0.000
            …

        Detection logic:
          1. If the first column header is a recognised wavenumber keyword
             (e.g. "wavenumber", "raman shift", "cm-1") → Format A.
          2. Otherwise attempt to parse the first column as floats → Format A.
          3. On failure (string labels in first column) → Format B.
        """
        path = filedialog.askopenfilename(
            title="Select Raman CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return   # User cancelled — do nothing

        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                messagebox.showerror("Error", "CSV must have at least 2 columns.")
                return

            first_header = str(df.columns[0]).lower().strip()
            # Known column header strings that indicate the wavenumber axis
            wn_keywords = {
                "wavenumber", "wave", "raman shift", "cm-1", "cm⁻¹",
                "shift", "wavenumbers", "raman_shift",
            }

            if first_header in wn_keywords:
                # Format A confirmed by header keyword
                wavenumbers = df.iloc[:, 0].values.astype(float)
                spectra     = df.iloc[:, 1:].values.T.astype(float)   # Transpose: rows→spectra
                labels      = list(df.columns[1:])                    # Column names = sample IDs
            else:
                # Try numeric first column → also Format A
                try:
                    wavenumbers = df.iloc[:, 0].values.astype(float)
                    spectra     = df.iloc[:, 1:].values.T.astype(float)
                    labels      = list(df.columns[1:])
                except (ValueError, TypeError):
                    # First column contains strings → Format B (transposed)
                    wavenumbers = np.array(df.columns[1:], dtype=float)    # Headers are wavenumbers
                    spectra     = df.iloc[:, 1:].values.astype(float)      # Each row is a spectrum
                    labels      = list(df.iloc[:, 0].astype(str))          # First column = labels

            # Sanity check: intensity points must equal wavenumber count
            if spectra.shape[1] != len(wavenumbers):
                messagebox.showerror(
                    "Shape mismatch",
                    f"{spectra.shape[1]} intensity points but {len(wavenumbers)} wavenumbers.",
                )
                return

            self._set_raman_data(wavenumbers, spectra, labels)

        except Exception as exc:
            messagebox.showerror("Error loading Raman CSV", str(exc))

    def _set_raman_data(self, wavenumbers, spectra, labels):
        """Store loaded Raman data and wire it into the PCA analysis pipeline.

        Converts the Raman matrix into the same format expected by _do_analysis:
          • data_X        → spectra matrix (n_spectra × n_wavenumbers)
          • data_y        → integer-encoded material labels
          • feature_names → wavenumber strings as column names
          • target_names  → unique material names for legend / stats display

        Also refreshes the data-preview tab, redraws the Raman viewer, and
        switches the notebook to the Raman tab.

        Parameters
        ----------
        wavenumbers : ndarray, shape (n_wn,)           — wavenumber axis in cm⁻¹
        spectra     : ndarray, shape (n_spectra, n_wn) — intensity matrix
        labels      : list[str], length n_spectra      — material name per spectrum
        """
        # ── Store Raman-specific state ─────────────────────────────────────────
        self.raman_wavenumbers = wavenumbers
        self.raman_spectra     = spectra
        self.raman_labels      = labels

        # ── Build label encoder (string → int) ───────────────────────────────
        unique_labels = sorted(set(labels))                          # Deterministic ordering
        label_to_int  = {l: i for i, l in enumerate(unique_labels)} # Map name → integer

        # ── Populate PCA-pipeline attributes ─────────────────────────────────
        col_names          = [f"{w:.1f}" for w in wavenumbers]       # "521.0", "1332.0", …
        self.data_X        = pd.DataFrame(spectra, columns=col_names)
        self.data_y        = np.array([label_to_int[l] for l in labels])
        self.feature_names = col_names
        self.target_names  = unique_labels

        n_samples, n_features = self.data_X.shape
        n_classes             = len(unique_labels)
        self.data_info_var.set(
            f"✓ Raman: {n_samples} spectra, {n_features} wavenumbers, {n_classes} classes"
        )
        # Choose a sensible default component count (≤ 3, but within data limits)
        self.n_components_var.set(min(3, n_samples - 1, n_features))

        self._show_data_preview()       # Refresh the 'Data Preview' tab text
        self._plot_raman_spectra()      # Redraw the Raman viewer
        self.notebook.select(self.tab_raman)   # Bring the Raman tab to the front

    def _plot_raman_spectra(self):
        """Redraw all spectra in the Raman viewer with current display options.

        Respects the normalize and stack toggles from the left control panel.

        Normalise  : divides each spectrum by its own maximum → max = 1.
        Stack      : adds a constant vertical offset per class so spectra in
                     different groups are separated vertically for clarity.

        Legend entries are de-duplicated: only the first occurrence of each
        material label is added (subsequent lines use "_nolegend_").
        """
        if self.raman_spectra is None:
            return   # Nothing loaded yet — leave the placeholder visible

        fig = self.raman_fig
        fig.clf()                           # Clear previous drawing
        ax  = fig.add_subplot(111)
        # Append a marker to the title when processed spectra are being displayed
        plot_title = "Raman Spectra" + (" [processed]" if self.raman_processed is not None else "")
        self._style_ax(ax, plot_title)

        wn = self.raman_wavenumbers
        # Use processed spectra (after noise reduction / coating) when available;
        # fall back to the raw matrix otherwise so display is always up-to-date.
        src     = self.raman_processed if self.raman_processed is not None else self.raman_spectra
        spectra = src.copy()   # Work on a copy so the stored arrays are not mutated
        labels  = self.raman_labels
        unique_labels = sorted(set(labels))

        # ── Optional: normalise each spectrum to its own maximum ──────────────
        if self.raman_normalize_var.get():
            mx          = spectra.max(axis=1, keepdims=True)   # Per-spectrum max, shape (n, 1)
            mx[mx == 0] = 1                                    # Avoid division by zero
            spectra     = spectra / mx

        # ── Optional: vertical stack offset ───────────────────────────────────
        # The offset per class equals 18 % of the global maximum intensity
        stack       = self.raman_stack_var.get()
        offset_step = spectra.max() * 0.18 if stack else 0.0

        seen = {}   # Track first occurrence of each label for the legend
        for spectrum, label in zip(spectra, labels):
            color      = PLOT_COLORS[unique_labels.index(label) % len(PLOT_COLORS)]
            y          = spectrum + offset_step * unique_labels.index(label)  # Class-based offset
            lbl        = label if label not in seen else "_nolegend_"          # Skip repeat legends
            line, = ax.plot(wn, y, color=color, alpha=0.65, linewidth=0.9, label=lbl)
            if label not in seen:
                seen[label] = line   # Remember the first line per label

        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER, fontsize=9, labelcolor=TEXT)

        # ── Vertical crosshair (invisible until hover) ────────────────────────
        # alpha=0.0 hides the line; _on_raman_hover raises it on mouse entry
        self._raman_vline = ax.axvline(
            x=wn[0], color=ACCENT, linewidth=1.0, linestyle="--", alpha=0.0,
        )
        self.raman_ax = ax       # Store reference so hover callback can check inaxes
        self.raman_canvas.draw()

    def _on_raman_hover(self, event):
        """Handle mouse-move events over the Raman axes.

        Updates the cursor-readout label and moves the vertical dashed crosshair
        to the current mouse x-position.  Uses draw_idle() (not draw()) to
        avoid excessive redraws on fast mouse movement.

        Hides the crosshair when the mouse leaves the axes.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
        """
        # Hide crosshair when mouse is outside the Raman axes
        if self.raman_ax is None or event.inaxes != self.raman_ax:
            if self._raman_vline is not None and self._raman_vline.get_alpha() > 0:
                self._raman_vline.set_alpha(0.0)     # Make crosshair invisible
                self.raman_canvas.draw_idle()        # Schedule redraw (non-blocking)
            return

        x, y = event.xdata, event.ydata   # Data-space coordinates of the mouse
        if x is None:
            return   # Can be None if the axes do not contain the cursor (edge case)

        # Update the text readout below the canvas
        self.raman_cursor_var.set(
            f"Wavenumber: {x:,.1f} cm⁻¹   |   Intensity: {y:.4f}"
        )
        # Move crosshair and make it visible
        if self._raman_vline is not None:
            self._raman_vline.set_xdata([x, x])    # x must be a sequence for axvline
            self._raman_vline.set_alpha(0.7)
            self.raman_canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Signal processing — noise reduction
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gaussian_kernel(sigma, size):
        """Return a normalised 1-D Gaussian convolution kernel.

        Used as a scipy-free fallback when scipy is not installed.

        Parameters
        ----------
        sigma : float — standard deviation in samples
        size  : int   — kernel length (should be odd)

        Returns
        -------
        ndarray, shape (size,) — normalised weights summing to 1
        """
        half  = size // 2
        x     = np.arange(-half, half + 1, dtype=float)   # Symmetric integer positions
        k     = np.exp(-x**2 / (2.0 * sigma**2))          # Un-normalised Gaussian
        return k / k.sum()                                 # Normalise so weights sum to 1

    def _apply_noise_reduction(self):
        """Apply the selected noise-reduction filter to the current Raman spectra.

        The filter is applied to raw spectra each time so repeated calls do not
        compound the smoothing.  The result is stored in ``self.raman_processed``
        and ``self.data_X`` is updated so that a subsequent PCA run uses the
        denoised data.

        Methods
        -------
        Savitzky-Golay : Polynomial least-squares fit in a sliding window.
                         Preserves peak positions and relative intensities well.
                         Requires scipy; window must be odd and ≥ polynomial order + 1.
        Gaussian       : Convolution with a Gaussian kernel.  Gentle, symmetric
                         smoothing; sigma ≈ window / 4.
        Moving Average : Uniform-weight convolution.  Fastest but can flatten peaks.
        Median         : Replaces each point with the median of its neighbourhood.
                         Excellent for impulsive noise / cosmic-ray spike removal.
        """
        if self.raman_spectra is None:
            messagebox.showwarning("No Data", "Load Raman data before applying noise reduction.")
            return

        method = self.noise_method_var.get()
        window = self.noise_window_var.get()
        window = max(3, window)                          # Minimum meaningful window
        if window % 2 == 0:
            window += 1                                  # Force odd (required by SG and median)

        src = self.raman_spectra   # Always start from the raw spectra (no cascading)

        if method == "Savitzky-Golay":
            if not _SCIPY_OK:
                messagebox.showerror(
                    "scipy required",
                    "Savitzky-Golay needs scipy.\n\npip install scipy",
                )
                return
            # polyorder=3 is a good balance between flexibility and smoothing
            result = np.array([_savgol_filter(s, window, polyorder=min(3, window - 1))
                                for s in src])

        elif method == "Gaussian":
            sigma = window / 4.0   # Approximate: ~95 % of energy within ±2σ ≈ window/2
            if _SCIPY_OK:
                result = np.array([_gfilt1d(s, sigma) for s in src])
            else:
                # Numpy fallback: explicit convolution with a Gaussian kernel
                kernel = self._gaussian_kernel(sigma, window)
                result = np.array([np.convolve(s, kernel, mode="same") for s in src])

        elif method == "Moving Average":
            kernel = np.ones(window) / window            # Uniform weights
            result = np.array([np.convolve(s, kernel, mode="same") for s in src])

        elif method == "Median":
            if _SCIPY_OK:
                result = np.array([_medfilt(s, window) for s in src])
            else:
                # Numpy fallback: manual sliding median
                half = window // 2
                result = []
                for s in src:
                    padded = np.pad(s, half, mode="edge")         # Reflect-pad edges
                    smoothed = np.array([
                        np.median(padded[i: i + window])
                        for i in range(len(s))
                    ])
                    result.append(smoothed)
                result = np.array(result)
        else:
            return   # Unknown method — should not happen

        self.raman_processed = np.clip(result, 0, None)   # Ensure non-negative values
        self._sync_processed_to_pca()                     # Update data_X for PCA pipeline
        self._plot_raman_spectra()                        # Redraw with [processed] title

    # ══════════════════════════════════════════════════════════════════════════
    # Signal processing — coating simulation
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_coating_simulation(self):
        """Simulate the effect of one or more coatings on the loaded Raman spectra.

        Physical model
        ──────────────
        A coating of thickness *t* and substrate-attenuation factor *a*:

          I_measured(ν) = I_substrate(ν) × (1 − a·t)  +  I_coating(ν) × t

        where I_coating is the sum of Lorentzian peaks defined in COATING_CATALOGUE
        for all checked coatings.  Multiple coatings are additive (mixed layer
        model — assumes incoherent scattering from all layers).

        The simulation always starts from the **raw** spectra so that multiple
        Apply calls with different settings produce independent results (no
        stacking of repeated coatings).

        Parameters exposed in the UI
        ────────────────────────────
        Coating thickness (0–1) : scales the coating peak amplitude.
                                  0 = coating absent; 1 = full coating contribution.
        Substrate attenuation (0–0.9) : fraction of substrate signal that is
                                        blocked / absorbed by the coating layer.
                                        Kept below 1.0 so the substrate is never
                                        completely obscured.
        """
        if self.raman_spectra is None:
            messagebox.showwarning("No Data", "Load Raman data before applying a coating simulation.")
            return

        active_coatings = [name for name, var in self._coating_vars.items() if var.get()]
        if not active_coatings:
            messagebox.showinfo("No Coating Selected", "Tick at least one coating to simulate.")
            return

        thickness   = self.coat_thickness_var.get()    # Coating thickness factor [0, 1]
        attenuation = self.coat_attenuation_var.get()  # Substrate signal reduction factor [0, 0.9]

        # ── Build the combined coating spectrum on the shared wavenumber axis ──
        coating_spectrum = np.zeros(len(RAMAN_WN))   # Accumulator; starts at zero
        for name in active_coatings:
            for center, width, amp in COATING_CATALOGUE[name]:
                # Each coating peak is modulated by the thickness slider
                coating_spectrum += _lorentzian(RAMAN_WN, center, width, amp) * thickness

        # ── Apply to every substrate spectrum ────────────────────────────────
        src    = self.raman_spectra   # Start from raw (not previously processed) spectra
        # Broadcast coating_spectrum (1-D) across all rows of src (2-D)
        result = src * (1.0 - attenuation * thickness) + coating_spectrum[np.newaxis, :]
        result = np.clip(result, 0, None)   # Physical constraint: no negative intensities

        self.raman_processed = result
        self._sync_processed_to_pca()      # Expose coated spectra to the PCA pipeline
        self._plot_raman_spectra()         # Redraw with [processed] title suffix

    def _reset_processing(self):
        """Discard processed spectra and restore the raw loaded data.

        Clears ``raman_processed`` so that ``_plot_raman_spectra`` reverts to
        the raw spectra, and syncs ``data_X`` back to the original intensities
        so that a subsequent PCA run uses unmodified measurements.
        """
        self.raman_processed = None      # Clear processed cache
        self._sync_processed_to_pca()   # Restore data_X from raw spectra
        self._plot_raman_spectra()       # Redraw without [processed] title

    def _sync_processed_to_pca(self):
        """Update ``data_X`` to reflect the currently active spectra matrix.

        Called after any processing step (noise reduction, coating, reset) to
        keep the PCA feature matrix in sync with what is displayed in the viewer.
        Uses processed spectra when available; raw spectra otherwise.
        """
        if self.raman_wavenumbers is None:
            return   # No Raman data loaded — nothing to sync
        src       = self.raman_processed if self.raman_processed is not None else self.raman_spectra
        col_names = [f"{w:.1f}" for w in self.raman_wavenumbers]
        self.data_X = pd.DataFrame(src, columns=col_names)   # Replace feature matrix in-place


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = PCAApp()    # Create and initialise the root window
    app.mainloop()    # Enter the Tk event loop (blocks until window is closed)
