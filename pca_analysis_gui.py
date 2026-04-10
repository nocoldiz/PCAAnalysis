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
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401 — registers the '3d' projection

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
    from scipy.signal import find_peaks as _find_peaks          # Local-maxima peak detector
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False   # scipy not installed; numpy-only fallbacks will be used


# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette  — Cosmic Indigo design system
# ═══════════════════════════════════════════════════════════════════════════════
BG           = "#07070f"   # Void-black root background
BG_PANEL     = "#0c0c1a"   # Left sidebar / header bar
BG_CARD      = "#111126"   # Card surfaces (sections, axes)
BG_ELEVATED  = "#1a1a35"   # Hover / active elevation
BORDER       = "#252545"   # Subtle dividers and spines
BORDER_HI    = "#3730a3"   # Accent border (focus ring)
TEXT         = "#f1f5f9"   # Primary text  (slate-100)
TEXT_DIM     = "#94a3b8"   # Secondary text (slate-400)
TEXT_MUTED   = "#475569"   # Tertiary / placeholder (slate-600)
INDIGO       = "#818cf8"   # Primary accent  (indigo-400)
INDIGO_DARK  = "#4f46e5"   # Pressed / deep accent
INDIGO_GLOW  = "#a5b4fc"   # Hover accent
VIOLET       = "#c084fc"   # Secondary accent (violet)
TEAL         = "#2dd4bf"   # Tertiary accent  (teal-400)
GREEN        = "#34d399"   # Success / positive (emerald-400)
RED          = "#f87171"   # Error / threshold  (red-400)
ORANGE       = "#fb923c"   # Warning            (orange-400)
AMBER        = "#fbbf24"   # Highlight          (amber-400)
CYAN         = "#22d3ee"   # Mono / data text   (cyan-400)
PURPLE       = "#c084fc"   # Purple alias (kept for compatibility)

# Backward-compat aliases used in plot helpers
ACCENT       = INDIGO
ACCENT_HOVER = INDIGO_GLOW
BG_SECONDARY = BG_PANEL

# Six perceptually distinct colours for scatter / Raman trace cycling
PLOT_COLORS = ["#f87171", "#34d399", "#818cf8", "#fbbf24", "#c084fc", "#22d3ee"]

# Decision-boundary colormaps (semi-transparent fills)
PLOT_CMAPS = [
    ListedColormap(["#fbbf24", "#f1f5f9", "#34d399"]),
    ListedColormap(["#f87171", "#dbeafe", "#a7f3d0"]),
]

# ── CPK element properties used by the 3-D molecular viewer ───────────────────
# Each entry: display colour (hex), covalent radius (Å), scatter marker size
ELEMENT_PROPS = {
    "H":  {"color": "#E8E8E8", "cov_r": 0.31, "size": 80},
    "C":  {"color": "#505050", "cov_r": 0.76, "size": 160},
    "N":  {"color": "#3050F8", "cov_r": 0.71, "size": 150},
    "O":  {"color": "#FF2010", "cov_r": 0.66, "size": 140},
    "F":  {"color": "#90E050", "cov_r": 0.57, "size": 120},
    "S":  {"color": "#E8E820", "cov_r": 1.05, "size": 200},
    "Cl": {"color": "#1FF01F", "cov_r": 0.99, "size": 190},
    "Br": {"color": "#A62929", "cov_r": 1.14, "size": 210},
    "P":  {"color": "#FF8000", "cov_r": 1.07, "size": 195},
    "Si": {"color": "#F0C8A0", "cov_r": 1.11, "size": 205},
    "Na": {"color": "#AB5CF2", "cov_r": 1.66, "size": 280},
    "K":  {"color": "#8F40D4", "cov_r": 2.03, "size": 320},
    "Ca": {"color": "#3DFF00", "cov_r": 1.76, "size": 290},
    "Mg": {"color": "#8AFF00", "cov_r": 1.41, "size": 240},
    "Ti": {"color": "#BFC2C7", "cov_r": 1.36, "size": 230},
    "Fe": {"color": "#E06633", "cov_r": 1.16, "size": 215},
    "Al": {"color": "#BFA6A6", "cov_r": 1.21, "size": 215},
    "Zn": {"color": "#7D80B0", "cov_r": 1.22, "size": 220},
    "Cu": {"color": "#C88033", "cov_r": 1.32, "size": 225},
}
_ELEM_DEFAULT = {"color": "#FF69B4", "cov_r": 1.50, "size": 200}  # Pink fallback


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

# ═══════════════════════════════════════════════════════════════════════════════
# 3-D Molecular structure database
# ═══════════════════════════════════════════════════════════════════════════════
# Atoms stored as (element_symbol, x_Å, y_Å, z_Å).
# Coordinates from standard references: NIST WebBook, CCDC, literature.
# Ionic / mineral fragments show the asymmetric unit or a small cluster.

MOLECULAR_DATABASE = {
    # ── Simple diatomics ─────────────────────────────────────────────────────
    "H2": {
        "name": "Hydrogen", "formula": "H₂", "category": "Diatomic",
        "description": "H–H single bond  (0.741 Å)",
        "atoms": [("H", -0.371, 0.000, 0.000), ("H", 0.371, 0.000, 0.000)],
    },
    "O2": {
        "name": "Oxygen", "formula": "O₂", "category": "Diatomic",
        "description": "O=O double bond  (1.208 Å)",
        "atoms": [("O", -0.604, 0.000, 0.000), ("O", 0.604, 0.000, 0.000)],
    },
    "N2": {
        "name": "Nitrogen", "formula": "N₂", "category": "Diatomic",
        "description": "N≡N triple bond  (1.098 Å)",
        "atoms": [("N", -0.549, 0.000, 0.000), ("N", 0.549, 0.000, 0.000)],
    },
    "HCl": {
        "name": "Hydrogen Chloride", "formula": "HCl", "category": "Diatomic",
        "description": "H–Cl bond  (1.274 Å)",
        "atoms": [("H", -0.637, 0.000, 0.000), ("Cl", 0.637, 0.000, 0.000)],
    },
    # ── Triatomics ────────────────────────────────────────────────────────────
    "H2O": {
        "name": "Water", "formula": "H₂O", "category": "Inorganic",
        "description": "Bent  H–O–H = 104.5°,  O–H = 0.957 Å",
        "atoms": [
            ("O",  0.000,  0.000, 0.000),
            ("H",  0.757,  0.586, 0.000),
            ("H", -0.757,  0.586, 0.000),
        ],
    },
    "CO2": {
        "name": "Carbon Dioxide", "formula": "CO₂", "category": "Inorganic",
        "description": "Linear  O=C=O,  C=O = 1.163 Å",
        "atoms": [
            ("O", -1.163, 0.000, 0.000),
            ("C",  0.000, 0.000, 0.000),
            ("O",  1.163, 0.000, 0.000),
        ],
    },
    "HCN": {
        "name": "Hydrogen Cyanide", "formula": "HCN", "category": "Inorganic",
        "description": "Linear  H–C≡N,  C–N = 1.156 Å",
        "atoms": [
            ("H", -1.065, 0.000, 0.000),
            ("C",  0.000, 0.000, 0.000),
            ("N",  1.156, 0.000, 0.000),
        ],
    },
    # ── Small inorganic polyatomics ───────────────────────────────────────────
    "NH3": {
        "name": "Ammonia", "formula": "NH₃", "category": "Inorganic",
        "description": "Trigonal pyramidal  H–N–H = 107.8°,  N–H = 1.012 Å",
        "atoms": [
            ("N",  0.000,  0.000,  0.000),
            ("H",  0.000, -0.938, -0.382),
            ("H",  0.812,  0.469, -0.382),
            ("H", -0.812,  0.469, -0.382),
        ],
    },
    "CH4": {
        "name": "Methane", "formula": "CH₄", "category": "Organic",
        "description": "Tetrahedral  Td,  C–H = 1.091 Å",
        "atoms": [
            ("C",  0.000,  0.000,  0.000),
            ("H",  0.630,  0.630,  0.630),
            ("H", -0.630, -0.630,  0.630),
            ("H", -0.630,  0.630, -0.630),
            ("H",  0.630, -0.630, -0.630),
        ],
    },
    "CH2O": {
        "name": "Formaldehyde", "formula": "CH₂O", "category": "Organic",
        "description": "Planar  C=O,  H–C–H = 116.5°",
        "atoms": [
            ("C",  0.000,  0.000, 0.000),
            ("O",  1.208,  0.000, 0.000),
            ("H", -0.590,  0.952, 0.000),
            ("H", -0.590, -0.952, 0.000),
        ],
    },
    # ── Organic molecules ─────────────────────────────────────────────────────
    "C2H2": {
        "name": "Acetylene", "formula": "C₂H₂", "category": "Organic",
        "description": "Linear  H–C≡C–H,  C–C = 1.203 Å",
        "atoms": [
            ("H", -1.667, 0.000, 0.000),
            ("C", -0.602, 0.000, 0.000),
            ("C",  0.602, 0.000, 0.000),
            ("H",  1.667, 0.000, 0.000),
        ],
    },
    "C2H4": {
        "name": "Ethylene", "formula": "C₂H₄", "category": "Organic",
        "description": "Planar  C=C = 1.339 Å,  H–C–H = 116.6°",
        "atoms": [
            ("C", -0.670,  0.000, 0.000),
            ("C",  0.670,  0.000, 0.000),
            ("H", -1.241,  0.924, 0.000),
            ("H", -1.241, -0.924, 0.000),
            ("H",  1.241,  0.924, 0.000),
            ("H",  1.241, -0.924, 0.000),
        ],
    },
    "C2H6": {
        "name": "Ethane", "formula": "C₂H₆", "category": "Organic",
        "description": "Staggered  C–C = 1.524 Å,  H–C–C = 111.2°",
        "atoms": [
            ("C", -0.762,  0.000,  0.000),
            ("C",  0.762,  0.000,  0.000),
            ("H", -1.157,  1.018,  0.000),
            ("H", -1.157, -0.509,  0.882),
            ("H", -1.157, -0.509, -0.882),
            ("H",  1.157, -1.018,  0.000),
            ("H",  1.157,  0.509,  0.882),
            ("H",  1.157,  0.509, -0.882),
        ],
    },
    "CH3OH": {
        "name": "Methanol", "formula": "CH₃OH", "category": "Organic",
        "description": "C–O = 1.431 Å,  O–H = 0.960 Å,  C–O–H = 108.5°",
        "atoms": [
            ("C",  0.000,  0.000,  0.000),
            ("O",  1.431,  0.000,  0.000),
            ("H",  1.737,  0.912,  0.000),
            ("H", -0.363,  1.029,  0.000),
            ("H", -0.363, -0.515,  0.890),
            ("H", -0.363, -0.515, -0.890),
        ],
    },
    "C2H5OH": {
        "name": "Ethanol", "formula": "C₂H₅OH", "category": "Organic",
        "description": "C–C–O backbone,  C–O = 1.431 Å,  anti conformation",
        "atoms": [
            ("C", -1.232,  0.082,  0.000),
            ("C",  0.000, -0.767,  0.000),
            ("O",  1.175,  0.073,  0.000),
            ("H",  1.972, -0.476,  0.000),
            ("H", -1.231,  0.720,  0.890),
            ("H", -1.231,  0.720, -0.890),
            ("H", -2.138, -0.530,  0.000),
            ("H",  0.000, -1.406,  0.890),
            ("H",  0.000, -1.406, -0.890),
        ],
    },
    "C3H8": {
        "name": "Propane", "formula": "C₃H₈", "category": "Organic",
        "description": "Extended chain  C–C–C = 112°,  C–C = 1.532 Å",
        "atoms": [
            ("C", -1.268,  0.000,  0.000),
            ("C",  0.000,  0.620,  0.000),
            ("C",  1.268,  0.000,  0.000),
            ("H", -1.900,  0.633,  0.629),
            ("H", -1.900,  0.633, -0.629),
            ("H", -1.450, -1.040,  0.000),
            ("H",  0.000,  1.258,  0.889),
            ("H",  0.000,  1.258, -0.889),
            ("H",  1.900,  0.633,  0.629),
            ("H",  1.900,  0.633, -0.629),
            ("H",  1.450, -1.040,  0.000),
        ],
    },
    "C6H6": {
        "name": "Benzene", "formula": "C₆H₆", "category": "Organic",
        "description": "Aromatic ring  D₆h,  C–C = 1.397 Å  (resonance)",
        "atoms": [
            ("C",  1.397,  0.000, 0.000),
            ("C",  0.699,  1.210, 0.000),
            ("C", -0.699,  1.210, 0.000),
            ("C", -1.397,  0.000, 0.000),
            ("C", -0.699, -1.210, 0.000),
            ("C",  0.699, -1.210, 0.000),
            ("H",  2.482,  0.000, 0.000),
            ("H",  1.241,  2.149, 0.000),
            ("H", -1.241,  2.149, 0.000),
            ("H", -2.482,  0.000, 0.000),
            ("H", -1.241, -2.149, 0.000),
            ("H",  1.241, -2.149, 0.000),
        ],
    },
    "H2SO4": {
        "name": "Sulfuric Acid", "formula": "H₂SO₄", "category": "Inorganic",
        "description": "Tetrahedral S  —  2 S=O + 2 S–OH,  S–O = 1.43 / 1.57 Å",
        "atoms": [
            ("S",  0.000,  0.000,  0.000),
            ("O",  1.052,  1.052,  0.450),
            ("O", -1.052,  1.052, -0.450),
            ("O",  0.900, -1.100,  0.000),
            ("O", -0.900, -1.100,  0.000),
            ("H",  1.750, -1.650,  0.000),
            ("H", -1.750, -1.650,  0.000),
        ],
    },
    # ── Mineral / crystal unit fragments ─────────────────────────────────────
    "SiO2": {
        "name": "Silicon Dioxide (Quartz)", "formula": "SiO₂", "category": "Mineral",
        "description": "SiO₄ tetrahedron  —  Si–O = 1.610 Å,  O–Si–O = 109.5°",
        "atoms": [
            ("Si",  0.000,  0.000,  0.000),
            ("O",   0.930,  0.930,  0.930),
            ("O",   0.930, -0.930, -0.930),
            ("O",  -0.930,  0.930, -0.930),
            ("O",  -0.930, -0.930,  0.930),
        ],
    },
    "CaCO3": {
        "name": "Calcium Carbonate (Calcite)", "formula": "CaCO₃", "category": "Mineral",
        "description": "Planar CO₃²⁻  +  Ca²⁺,  C–O = 1.290 Å",
        "atoms": [
            ("C",   0.000,  0.000,  0.000),
            ("O",   1.290,  0.000,  0.000),
            ("O",  -0.645,  1.117,  0.000),
            ("O",  -0.645, -1.117,  0.000),
            ("Ca",  0.000,  0.000,  2.360),
        ],
    },
    "TiO2": {
        "name": "Titanium Dioxide (Anatase)", "formula": "TiO₂", "category": "Mineral",
        "description": "TiO₆ octahedron  —  anatase phase,  Ti–O = 1.93 / 1.97 Å",
        "atoms": [
            ("Ti",  0.000,  0.000,  0.000),
            ("O",   1.930,  0.000,  0.000),
            ("O",  -1.930,  0.000,  0.000),
            ("O",   0.000,  1.930,  0.000),
            ("O",   0.000, -1.930,  0.000),
            ("O",   0.000,  0.000,  1.970),
            ("O",   0.000,  0.000, -1.970),
        ],
    },
    "NaCl": {
        "name": "Sodium Chloride (Rock Salt)", "formula": "NaCl", "category": "Mineral",
        "description": "FCC ionic crystal cluster  —  d(Na–Cl) = 2.82 Å",
        "atoms": [
            ("Na",  0.000,  0.000,  0.000),
            ("Na",  0.000,  2.820,  2.820),
            ("Na",  2.820,  0.000,  2.820),
            ("Na",  2.820,  2.820,  0.000),
            ("Cl",  2.820,  0.000,  0.000),
            ("Cl",  0.000,  2.820,  0.000),
            ("Cl",  0.000,  0.000,  2.820),
            ("Cl",  2.820,  2.820,  2.820),
        ],
    },
    "Al2O3": {
        "name": "Aluminium Oxide (Corundum)", "formula": "Al₂O₃", "category": "Mineral",
        "description": "α-corundum AlO₆ octahedral pair  —  Al–O = 1.86 / 1.97 Å",
        "atoms": [
            ("Al",  0.000,  0.000,  0.000),
            ("Al",  0.000,  0.000,  2.165),
            ("O",   1.856,  0.000,  0.548),
            ("O",  -0.928,  1.608,  0.548),
            ("O",  -0.928, -1.608,  0.548),
            ("O",   1.856,  0.000,  1.617),
            ("O",  -0.928,  1.608,  1.617),
            ("O",  -0.928, -1.608,  1.617),
        ],
    },
}

# Flat list used to populate the molecule preset combobox (sorted by category then name)
_MOL_KEYS_SORTED = sorted(
    MOLECULAR_DATABASE.keys(),
    key=lambda k: (MOLECULAR_DATABASE[k]["category"], MOLECULAR_DATABASE[k]["name"]),
)


def _compute_bonds(atoms):
    """Return a list of (i, j) index pairs for atoms that are covalently bonded.

    Bonding criterion: distance ≤ (r_cov_i + r_cov_j) × 1.15
    where r_cov is the covalent radius from ELEMENT_PROPS.

    Parameters
    ----------
    atoms : list of (element, x, y, z)

    Returns
    -------
    list of (int, int) — index pairs
    """
    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ei, xi, yi, zi = atoms[i]
            ej, xj, yj, zj = atoms[j]
            ri = ELEMENT_PROPS.get(ei, _ELEM_DEFAULT)["cov_r"]
            rj = ELEMENT_PROPS.get(ej, _ELEM_DEFAULT)["cov_r"]
            d  = ((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2) ** 0.5
            if d <= (ri + rj) * 1.15:
                bonds.append((i, j))
    return bonds


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
        self.raman_labels      = None
        self.raman_ax          = None
        self._raman_vline      = None
        self._coating_vars     = {}
        self._last_mol_key     = None   # tracks last rendered molecule for export

        # ── UI status ─────────────────────────────────────────────────────────
        self.status_var   = tk.StringVar(value="Ready")
        self._hover_btns  = []   # (widget, normal_bg) for hover restore

        self._build_styles()
        self._build_layout()

    # ══════════════════════════════════════════════════════════════════════════
    # Styling
    # ══════════════════════════════════════════════════════════════════════════

    def _build_styles(self):
        """Configure ttk Style rules for the Cosmic Indigo theme."""
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame",       background=BG)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("Card.TFrame",  background=BG_CARD)

        style.configure("TLabel",         background=BG,       foreground=TEXT,     font=("Segoe UI", 10))
        style.configure("Panel.TLabel",   background=BG_PANEL, foreground=TEXT,     font=("Segoe UI", 10))
        style.configure("Card.TLabel",    background=BG_CARD,  foreground=TEXT,     font=("Segoe UI", 10))
        style.configure("Header.TLabel",  background=BG_PANEL, foreground=INDIGO,   font=("Segoe UI", 13, "bold"))
        style.configure("Dim.TLabel",     background=BG_CARD,  foreground=TEXT_DIM, font=("Segoe UI", 9))
        style.configure("PanelDim.TLabel",background=BG_PANEL, foreground=TEXT_DIM, font=("Segoe UI", 9))
        style.configure("Stat.TLabel",    background=BG_CARD,  foreground=CYAN,     font=("Consolas", 9))

        # Buttons
        style.configure("TButton",       background=BG_ELEVATED, foreground=TEXT,  font=("Segoe UI", 9), padding=(10, 5), borderwidth=0, relief="flat")
        style.map("TButton",             background=[("active", BG_ELEVATED), ("pressed", BORDER_HI)])
        style.configure("Accent.TButton",background=INDIGO_DARK, foreground=TEXT,  font=("Segoe UI", 10, "bold"), padding=(14, 8), borderwidth=0, relief="flat")
        style.map("Accent.TButton",      background=[("active", INDIGO), ("disabled", BORDER)], foreground=[("disabled", TEXT_MUTED)])

        # Combobox / Spinbox
        style.configure("TCombobox",     fieldbackground=BG_ELEVATED, background=BG_CARD, foreground=TEXT, selectbackground=INDIGO_DARK, selectforeground=TEXT, arrowcolor=TEXT_DIM)
        style.map("TCombobox",           fieldbackground=[("readonly", BG_ELEVATED)])
        style.configure("TSpinbox",      fieldbackground=BG_ELEVATED, background=BG_CARD, foreground=TEXT, arrowcolor=TEXT_DIM)

        # Checkbutton
        style.configure("TCheckbutton",  background=BG_CARD, foreground=TEXT_DIM,  font=("Segoe UI", 9), indicatorbackground=BG_ELEVATED, indicatorforeground=INDIGO)
        style.map("TCheckbutton",        background=[("active", BG_CARD)], foreground=[("active", TEXT)])

        # Notebook — sleek pill-style tabs
        style.configure("TNotebook",     background=BG_PANEL, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=TEXT_MUTED, padding=(16, 7), font=("Segoe UI", 9), borderwidth=0)
        style.map("TNotebook.Tab",       background=[("selected", BG_CARD)], foreground=[("selected", INDIGO)], font=[("selected", ("Segoe UI", 9, "bold"))])

        # Scrollbar
        style.configure("TScrollbar",    background=BG_ELEVATED, troughcolor=BG_PANEL, arrowcolor=TEXT_MUTED, borderwidth=0, relief="flat")

        # Progress bar
        style.configure("Horizontal.TProgressbar", background=INDIGO, troughcolor=BG_ELEVATED, borderwidth=0)

    # ══════════════════════════════════════════════════════════════════════════
    # Layout construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        """Build root structure: header bar → split pane → status bar."""

        # ── Header bar ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_PANEL, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Logo
        logo = tk.Frame(hdr, bg=BG_PANEL)
        logo.pack(side="left", padx=18, pady=0)
        tk.Label(logo, text="◆", bg=BG_PANEL, fg=INDIGO,
                 font=("Segoe UI", 20, "bold")).pack(side="left", pady=10)
        tk.Label(logo, text="  PCA Analysis Tool", bg=BG_PANEL, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(logo, text="  v2.0", bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="left", pady=(12, 0))

        # Export buttons (right side of header)
        exp = tk.Frame(hdr, bg=BG_PANEL)
        exp.pack(side="right", padx=16)
        tk.Label(exp, text="EXPORT", bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 7, "bold")).pack(side="left", padx=(0, 6), pady=18)
        for label, cmd, fg in [
            ("⬡  PDF",  self._export_pdf,  VIOLET),
            ("⬡  CSV",  self._export_csv,  TEAL),
            ("⬡  PNG",  self._export_png,  GREEN),
        ]:
            b = tk.Button(exp, text=label, command=cmd, bg=BG_CARD, fg=fg,
                          font=("Segoe UI", 9, "bold"), relief="flat",
                          padx=14, pady=5, cursor="hand2", bd=0,
                          activebackground=BG_ELEVATED, activeforeground=fg)
            b.pack(side="left", padx=3, pady=12)

        # ── Accent separator ──────────────────────────────────────────────────
        tk.Frame(self, bg=INDIGO_DARK, height=1).pack(fill="x")

        # ── Status bar (bottom) ───────────────────────────────────────────────
        sbar = tk.Frame(self, bg=BG_PANEL, height=26)
        sbar.pack(side="bottom", fill="x")
        sbar.pack_propagate(False)
        tk.Frame(sbar, bg=INDIGO_DARK, width=2).pack(side="left", fill="y")
        self.status_label = tk.Label(
            sbar, textvariable=self.status_var,
            bg=BG_PANEL, fg=TEXT_MUTED,
            font=("Consolas", 8), padx=10, anchor="w",
        )
        self.status_label.pack(side="left", fill="y")
        tk.Label(sbar, text="Cosmic Indigo  ◆", bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), padx=12).pack(side="right")

        # ── Main paned area ───────────────────────────────────────────────────
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=BG_PANEL, width=328)
        main.add(left, weight=0)
        self._build_controls(left)

        right = ttk.Frame(main)
        main.add(right, weight=1)
        self._build_results(right)

    # ══════════════════════════════════════════════════════════════════════════
    # Control panel helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _card(self, parent, title, icon="", color=None):
        """Return a content frame inside a styled card with a colored left stripe.

        Creates:  [3px stripe | header row | content frame]
        The card lives inside *parent* (the scrollable panel).
        """
        color = color or INDIGO
        wrap = tk.Frame(parent, bg=BG_PANEL)
        wrap.pack(fill="x", padx=8, pady=(0, 6))
        tk.Frame(wrap, bg=color, width=3).pack(side="left", fill="y")
        body = tk.Frame(wrap, bg=BG_CARD)
        body.pack(side="left", fill="both", expand=True)
        if title:
            hf = tk.Frame(body, bg=BG_CARD)
            hf.pack(fill="x", padx=10, pady=(7, 0))
            lbl_text = f"{icon}  {title}" if icon else title
            tk.Label(hf, text=lbl_text, bg=BG_CARD, fg=color,
                     font=("Segoe UI", 8, "bold")).pack(side="left")
        content = tk.Frame(body, bg=BG_CARD)
        content.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        return content

    def _flat_btn(self, parent, text, command, fg=None, bg=None):
        """Return a flat tk.Button styled for the card theme."""
        fg  = fg  or TEXT_DIM
        bg  = bg  or BG_ELEVATED
        btn = tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg, activebackground=BG_ELEVATED, activeforeground=fg,
            relief="flat", bd=0, font=("Segoe UI", 9),
            padx=10, pady=5, cursor="hand2",
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=BG_ELEVATED if bg == BG_CARD else BG_CARD))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    def _lbl(self, parent, text, dim=False):
        """Compact label factory for card interiors."""
        fg = TEXT_DIM if dim else TEXT
        return tk.Label(parent, text=text, bg=BG_CARD, fg=fg, font=("Segoe UI", 9), anchor="w")

    def _combo(self, parent, var, values, width=28):
        """Return a ttk.Combobox with card styling."""
        c = ttk.Combobox(parent, textvariable=var, state="readonly",
                         width=width, values=values)
        return c

    def _scale(self, parent, var, from_, to, res):
        return tk.Scale(
            parent, variable=var, from_=from_, to=to, resolution=res,
            orient="horizontal", bg=BG_CARD, fg=TEXT_DIM,
            troughcolor=BG_ELEVATED, highlightthickness=0,
            sliderrelief="flat", activebackground=INDIGO,
            font=("Consolas", 8), showvalue=True,
        )

    def _check(self, parent, text, var, command=None):
        kw = {"command": command} if command else {}
        return ttk.Checkbutton(parent, text=text, variable=var,
                               style="TCheckbutton", **kw)

    # ══════════════════════════════════════════════════════════════════════════
    # Control panel build
    # ══════════════════════════════════════════════════════════════════════════

    def _build_controls(self, parent):
        """Build the scrollable left panel with card-based sections."""
        # Scrollable canvas
        cv = tk.Canvas(parent, bg=BG_PANEL, highlightthickness=0, width=328)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        sf = tk.Frame(cv, bg=BG_PANEL)
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=sf, anchor="nw", width=316)
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(int(-1 * e.delta / 120), "units"))

        p = sf  # alias

        # ── Top spacer ────────────────────────────────────────────────────────
        tk.Frame(p, bg=BG_PANEL, height=8).pack(fill="x")

        # ══ CARD: DATA SOURCE ════════════════════════════════════════════════
        c = self._card(p, "DATA SOURCE", "◈", INDIGO)

        self._lbl(c, "Dataset preset").pack(anchor="w", pady=(0, 3))
        self.preset_var = tk.StringVar(value="── select preset ──")
        cb = self._combo(c, self.preset_var, [
            "── select preset ──",
            "Iris  (4 features · 3 classes)",
            "Wine  (13 features · 3 classes)",
            "Breast Cancer  (30 features · 2 classes)",
            "Random Blobs  (5 features · 4 classes)",
            "Random Circles  (2 features · 2 classes)",
        ])
        cb.pack(fill="x", pady=(0, 6))
        cb.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self._flat_btn(c, "  Load CSV file", self._load_csv,
                       fg=INDIGO, bg=BG_CARD).pack(fill="x")

        self.data_info_var = tk.StringVar(value="No data loaded")
        tk.Label(c, textvariable=self.data_info_var, bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), anchor="w", wraplength=260).pack(anchor="w", pady=(6, 0))

        # ══ CARD: PCA PARAMETERS ════════════════════════════════════════════
        c = self._card(p, "PCA PARAMETERS", "◈", TEAL)

        row = tk.Frame(c, bg=BG_CARD)
        row.pack(fill="x", pady=(0, 6))
        self._lbl(row, "Components").pack(side="left")
        self.n_components_var = tk.IntVar(value=2)
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.n_components_var,
                    width=5, style="TSpinbox").pack(side="right")

        self._lbl(c, "Test split ratio").pack(anchor="w", pady=(2, 2))
        self.test_ratio_var = tk.DoubleVar(value=0.2)
        self._scale(c, self.test_ratio_var, 0.05, 0.5, 0.05).pack(fill="x")

        self.standardize_var = tk.BooleanVar(value=True)
        self._check(c, "Standardize features  (recommended)",
                    self.standardize_var).pack(anchor="w", pady=(6, 2))

        self.classify_var = tk.BooleanVar(value=True)
        self._check(c, "Run Logistic Regression classifier",
                    self.classify_var).pack(anchor="w")

        # ══ CARD: RAMAN DATA ════════════════════════════════════════════════
        c = self._card(p, "RAMAN DATA", "◈", VIOLET)

        self._lbl(c, "Spectral preset").pack(anchor="w", pady=(0, 3))
        self.raman_preset_var = tk.StringVar(value="── select preset ──")
        rcb = self._combo(c, self.raman_preset_var,
                          ["── select preset ──"] + list(RAMAN_PRESETS.keys()))
        rcb.pack(fill="x", pady=(0, 6))
        rcb.bind("<<ComboboxSelected>>", self._on_raman_preset_selected)

        self._flat_btn(c, "  Load Raman CSV", self._load_raman_csv,
                       fg=VIOLET, bg=BG_CARD).pack(fill="x", pady=(0, 8))

        self.raman_normalize_var = tk.BooleanVar(value=False)
        self._check(c, "Normalize spectra  (max = 1)",
                    self.raman_normalize_var,
                    command=self._plot_raman_spectra).pack(anchor="w", pady=(0, 2))

        self.raman_stack_var = tk.BooleanVar(value=False)
        self._check(c, "Stack with vertical offset",
                    self.raman_stack_var,
                    command=self._plot_raman_spectra).pack(anchor="w", pady=(0, 6))

        self._flat_btn(c, "  Identify Crystal Structure",
                       self._identify_crystal_structure,
                       fg=AMBER, bg=BG_CARD).pack(fill="x")

        # ══ CARD: SIGNAL PROCESSING ═════════════════════════════════════════
        c = self._card(p, "SIGNAL PROCESSING", "◈", ORANGE)

        self._lbl(c, "Noise reduction method", dim=True).pack(anchor="w", pady=(0, 2))
        self.noise_method_var = tk.StringVar(value="Savitzky-Golay")
        self._combo(c, self.noise_method_var, NOISE_METHODS).pack(fill="x", pady=(0, 4))

        wr = tk.Frame(c, bg=BG_CARD)
        wr.pack(fill="x", pady=(0, 4))
        self._lbl(wr, "Window size").pack(side="left")
        self.noise_window_var = tk.IntVar(value=11)
        ttk.Spinbox(wr, from_=3, to=101, increment=2,
                    textvariable=self.noise_window_var, width=5).pack(side="right")

        self._flat_btn(c, "  Apply Noise Reduction",
                       self._apply_noise_reduction,
                       fg=ORANGE, bg=BG_CARD).pack(fill="x", pady=(0, 8))

        # Separator
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(0, 8))

        self._lbl(c, "Coating simulation", dim=True).pack(anchor="w", pady=(0, 4))
        for coat_name in COATING_CATALOGUE:
            var = tk.BooleanVar(value=False)
            self._coating_vars[coat_name] = var
            self._check(c, coat_name, var).pack(anchor="w", pady=1)

        self._lbl(c, "Thickness").pack(anchor="w", pady=(8, 2))
        self.coat_thickness_var = tk.DoubleVar(value=0.3)
        self._scale(c, self.coat_thickness_var, 0.0, 1.0, 0.05).pack(fill="x")

        self._lbl(c, "Substrate attenuation").pack(anchor="w", pady=(6, 2))
        self.coat_attenuation_var = tk.DoubleVar(value=0.2)
        self._scale(c, self.coat_attenuation_var, 0.0, 0.9, 0.05).pack(fill="x")

        br = tk.Frame(c, bg=BG_CARD)
        br.pack(fill="x", pady=(8, 0))
        self._flat_btn(br, "  Apply Coating",
                       self._apply_coating_simulation,
                       fg=ORANGE, bg=BG_CARD).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._flat_btn(br, "⟳ Reset",
                       self._reset_processing,
                       fg=TEXT_DIM, bg=BG_CARD).pack(side="left", fill="x", expand=True)

        # ══ RUN BUTTON ═══════════════════════════════════════════════════════
        tk.Frame(p, bg=BG_PANEL, height=4).pack(fill="x")
        run_wrap = tk.Frame(p, bg=BG_PANEL)
        run_wrap.pack(fill="x", padx=8, pady=(0, 6))

        self.run_btn = tk.Button(
            run_wrap, text="▶   Run PCA Analysis",
            command=self._run_analysis,
            bg=INDIGO_DARK, fg=TEXT, activebackground=INDIGO,
            activeforeground=TEXT, relief="flat", bd=0,
            font=("Segoe UI", 11, "bold"), padx=16, pady=10, cursor="hand2",
        )
        self.run_btn.pack(fill="x")
        self.progress = ttk.Progressbar(run_wrap, mode="indeterminate",
                                        style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(4, 0))

        # ══ CARD: RESULTS SUMMARY ════════════════════════════════════════════
        c = self._card(p, "RESULTS SUMMARY", "◈", CYAN)
        self.stats_text = tk.Text(
            c, height=15,
            bg=BG_ELEVATED, fg=CYAN,
            font=("Consolas", 8), relief="flat", bd=0,
            insertbackground=CYAN, selectbackground=INDIGO_DARK,
            wrap="word", padx=8, pady=8,
        )
        self.stats_text.pack(fill="both", expand=True)
        self.stats_text.insert("1.0", "Run an analysis to see results here…")
        self.stats_text.config(state="disabled")

        tk.Frame(p, bg=BG_PANEL, height=12).pack(fill="x")

    # ══════════════════════════════════════════════════════════════════════════
    # Results area (right panel)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_results(self, parent):
        """Build the notebook with all result tabs."""
        # Tab strip background
        tk.Frame(parent, bg=BG_PANEL, height=1).pack(fill="x")
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.tab_raman    = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_molecule = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_scatter  = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_variance = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_decision = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_heatmap  = ttk.Frame(self.notebook, style="Card.TFrame")
        self.tab_data     = ttk.Frame(self.notebook, style="Card.TFrame")

        self.notebook.add(self.tab_raman,    text="  ◎ Raman  ")
        self.notebook.add(self.tab_molecule, text="  ◎ 3D Molecule  ")
        self.notebook.add(self.tab_scatter,  text="  ◎ PCA Scatter  ")
        self.notebook.add(self.tab_variance, text="  ◎ Variance  ")
        self.notebook.add(self.tab_decision, text="  ◎ Decision Boundary  ")
        self.notebook.add(self.tab_heatmap,  text="  ◎ Heatmap  ")
        self.notebook.add(self.tab_data,     text="  ◎ Data Preview  ")

        self._build_raman_tab(self.tab_raman)
        self._build_molecule_tab(self.tab_molecule)

        # ── matplotlib figures for PCA tabs ───────────────────────────────────
        self.figures  = {}
        self.canvases = {}

        for name, tab in [
            ("scatter",  self.tab_scatter),
            ("variance", self.tab_variance),
            ("decision", self.tab_decision),
            ("heatmap",  self.tab_heatmap),
        ]:
            fig = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab)
            toolbar.config(background=BG_ELEVATED)
            for child in toolbar.winfo_children():
                try:
                    child.config(background=BG_ELEVATED, foreground=TEXT_DIM)
                except Exception:
                    pass
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name]  = fig
            self.canvases[name] = canvas

        # ── Data preview tab ──────────────────────────────────────────────────
        self.data_text = scrolledtext.ScrolledText(
            self.tab_data,
            bg=BG_ELEVATED, fg=TEXT, font=("Consolas", 9),
            relief="flat", bd=0,
            insertbackground=TEXT, selectbackground=INDIGO_DARK,
        )
        self.data_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Placeholder text on each PCA plot
        for name, fig in self.figures.items():
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)
            ax.text(0.5, 0.5, "Load data and run analysis",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color=TEXT_MUTED, style="italic",
                    fontfamily="Segoe UI")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            self.canvases[name].draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Shared UI helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _section(self, parent, title):
        """Legacy section divider — kept for compatibility; prefer _card()."""
        if title:
            f = tk.Frame(parent, bg=BG_PANEL)
            f.pack(fill="x", padx=8, pady=(10, 2))
            tk.Label(f, text=title, bg=BG_PANEL, fg=TEXT_MUTED,
                     font=("Segoe UI", 8, "bold")).pack(side="left")
            tk.Frame(f, height=1, bg=BORDER).pack(side="left", fill="x",
                                                   expand=True, padx=(8, 0), pady=1)

    def _style_ax(self, ax, title=""):
        """Apply the Cosmic Indigo theme to a matplotlib Axes."""
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        if title:
            ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold",
                         pad=10, fontfamily="Segoe UI")
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)

    def _set_status(self, msg, color=None):
        """Update the bottom status bar message."""
        self.status_var.set(msg)
        if hasattr(self, "status_label"):
            self.status_label.config(foreground=color or TEXT_DIM)

    def _update_stats(self, text):
        """Replace content in the results summary text box."""
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    # Export — PDF / CSV / PNG
    # ══════════════════════════════════════════════════════════════════════════

    def _styled_toolbar(self, canvas, parent):
        """Attach and style a NavigationToolbar2Tk to *canvas*."""
        tb = NavigationToolbar2Tk(canvas, parent)
        tb.config(background=BG_ELEVATED)
        for ch in tb.winfo_children():
            try:
                ch.config(background=BG_ELEVATED, foreground=TEXT_DIM)
            except Exception:
                pass
        tb.update()
        return tb

    def _export_pdf(self):
        """Export all plots and a summary cover page to a PDF report."""
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF report", "*.pdf"), ("All files", "*.*")],
            title="Export PDF Report",
        )
        if not path:
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import datetime
            with PdfPages(path) as pdf:
                # Cover page
                fig_c = Figure(figsize=(8.27, 11.69), facecolor=BG_CARD)
                ax_c  = fig_c.add_subplot(111)
                ax_c.set_facecolor(BG_CARD)
                ax_c.set_axis_off()
                ax_c.text(0.5, 0.62, "◆  PCA Analysis Tool",
                          ha="center", fontsize=28, fontweight="bold",
                          color=INDIGO, transform=ax_c.transAxes)
                ax_c.text(0.5, 0.54, "Raman Spectroscopy · PCA · 3D Molecules",
                          ha="center", fontsize=13, color=TEXT_DIM,
                          transform=ax_c.transAxes)
                ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
                ax_c.text(0.5, 0.46, f"Generated  {ts}",
                          ha="center", fontsize=10, color=TEXT_MUTED,
                          transform=ax_c.transAxes)
                if self.data_X is not None:
                    info = (f"{self.data_X.shape[0]} samples · "
                            f"{self.data_X.shape[1]} features · "
                            f"{len(self.target_names or [])} classes")
                    ax_c.text(0.5, 0.40, info,
                              ha="center", fontsize=10, color=TEAL,
                              transform=ax_c.transAxes)
                pdf.savefig(fig_c, facecolor=BG_CARD)
                plt.close(fig_c)
                # Raman
                if self.raman_spectra is not None:
                    pdf.savefig(self.raman_fig, facecolor=BG_CARD)
                # Molecule
                if hasattr(self, "mol_fig") and self._last_mol_key:
                    pdf.savefig(self.mol_fig, facecolor=BG_CARD)
                # PCA plots
                for name in ["scatter", "variance", "decision", "heatmap"]:
                    if name in self.figures and self.pca_result is not None:
                        pdf.savefig(self.figures[name], facecolor=BG_CARD)
            self._set_status(f"PDF saved → {path}", GREEN)
            messagebox.showinfo("Export complete", f"PDF report saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self._set_status("PDF export failed", RED)

    def _export_csv(self):
        """Export the current feature matrix + labels as a CSV file."""
        if self.data_X is None:
            messagebox.showwarning("No data", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            title="Export Data as CSV",
        )
        if not path:
            return
        try:
            df = self.data_X.copy()
            if self.data_y is not None and self.target_names:
                df.insert(0, "label", [
                    self.target_names[i] if i < len(self.target_names) else str(i)
                    for i in self.data_y
                ])
            df.to_csv(path, index=False)
            self._set_status(f"CSV saved → {path}", GREEN)
            messagebox.showinfo("Export complete",
                                f"CSV saved ({df.shape[0]} rows × {df.shape[1]} cols):\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self._set_status("CSV export failed", RED)

    def _export_png(self):
        """Export every visible plot as a high-DPI PNG into a chosen folder."""
        import os
        folder = filedialog.askdirectory(title="Select PNG export folder")
        if not folder:
            return
        try:
            saved = []
            kw = dict(dpi=180, bbox_inches="tight")
            if self.raman_spectra is not None:
                p = os.path.join(folder, "raman_spectra.png")
                self.raman_fig.savefig(p, facecolor=BG_CARD, **kw)
                saved.append("raman_spectra.png")
            if hasattr(self, "mol_fig") and self._last_mol_key:
                p = os.path.join(folder, "molecule_3d.png")
                self.mol_fig.savefig(p, facecolor=BG_CARD, **kw)
                saved.append("molecule_3d.png")
            for name in ["scatter", "variance", "decision", "heatmap"]:
                if name in self.figures and self.pca_result is not None:
                    fname = f"pca_{name}.png"
                    self.figures[name].savefig(
                        os.path.join(folder, fname), facecolor=BG_CARD, **kw)
                    saved.append(fname)
            if not saved:
                messagebox.showinfo("Nothing to export",
                                    "Run an analysis or load Raman data first.")
                return
            self._set_status(f"{len(saved)} PNGs saved → {folder}", GREEN)
            messagebox.showinfo("Export complete",
                                f"Saved {len(saved)} images to:\n{folder}\n\n"
                                + "\n".join(saved))
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self._set_status("PNG export failed", RED)

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
            self.after(0, lambda: self._set_status(
                f"PCA complete — {n_comp} components, "
                f"{len(y_train)}/{len(y_test)} train/test", GREEN))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.after(0, lambda: self._set_status(f"Analysis error: {e}", RED))
        finally:
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.run_btn.config(
                state="normal", bg=INDIGO_DARK))

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
        ctrl = tk.Frame(parent, bg=BG_CARD)
        ctrl.pack(fill="x", padx=0, pady=0)
        tk.Label(ctrl, text="  ◎  Raman Spectra Viewer", bg=BG_CARD,
                 fg=VIOLET, font=("Segoe UI", 12, "bold")).pack(side="left", pady=8, padx=10)

        # ── matplotlib figure embedded in the Tk frame ────────────────────────
        self.raman_fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_CARD)
        self.raman_fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10)
        self.raman_canvas = FigureCanvasTkAgg(self.raman_fig, master=parent)

        toolbar = self._styled_toolbar(self.raman_canvas, parent)

        self.raman_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Cursor readout ────────────────────────────────────────────────────
        self.raman_cursor_var = tk.StringVar(
            value="Hover over the spectrum to read wavenumber and intensity"
        )
        tk.Label(parent, textvariable=self.raman_cursor_var,
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Consolas", 8)).pack(pady=(2, 6))

        self.raman_canvas.mpl_connect("motion_notify_event", self._on_raman_hover)

        # Placeholder axes
        ax = self.raman_fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        ax.text(0.5, 0.5, "Select a Raman preset or load a CSV to view spectra",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color=TEXT_MUTED, style="italic",
                fontfamily="Segoe UI")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.raman_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # 3-D Molecular viewer tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_molecule_tab(self, parent):
        """Build the interactive 3-D molecular structure viewer tab.

        Layout
        ------
        • Top control strip: formula entry, Build button, preset combobox,
          label toggle.
        • Centre: matplotlib 3-D Figure with NavigationToolbar (rotate/zoom/pan).
        • Bottom: one-line info label (atom count, bond count).
        """
        print("[Molecule] Building molecule tab")

        # ── Control strip ──────────────────────────────────────────────────────
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(ctrl, text="3D Molecular Viewer", style="Header.TLabel").pack(side="left")

        # Formula / name entry
        entry_frame = ttk.Frame(parent)
        entry_frame.pack(fill="x", padx=8, pady=(0, 2))

        ttk.Label(entry_frame, text="Formula / Name:").pack(side="left")
        self.mol_formula_var = tk.StringVar(value="H2O")
        mol_entry = ttk.Entry(entry_frame, textvariable=self.mol_formula_var, width=14)
        mol_entry.pack(side="left", padx=(4, 6))
        mol_entry.bind("<Return>", lambda e: self._build_molecule())

        ttk.Button(entry_frame, text="Build 3D",
                   command=self._build_molecule).pack(side="left", padx=(0, 12))

        ttk.Label(entry_frame, text="or preset:", style="Dim.TLabel").pack(side="left")
        self.mol_preset_var = tk.StringVar(value="— select —")
        preset_values = [f"{k}  —  {MOLECULAR_DATABASE[k]['name']}"
                         for k in _MOL_KEYS_SORTED]
        mol_combo = ttk.Combobox(
            entry_frame, textvariable=self.mol_preset_var,
            state="readonly", width=34, values=["— select —"] + preset_values,
        )
        mol_combo.pack(side="left", padx=(4, 0))
        mol_combo.bind("<<ComboboxSelected>>", self._on_mol_preset_selected)

        # Label toggle
        self.mol_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            entry_frame, text="Labels",
            variable=self.mol_labels_var,
            command=self._refresh_molecule,
        ).pack(side="left", padx=(10, 0))

        # ── 3-D matplotlib figure ──────────────────────────────────────────────
        self.mol_fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_CARD)
        self.mol_canvas = FigureCanvasTkAgg(self.mol_fig, master=parent)

        mol_toolbar = NavigationToolbar2Tk(self.mol_canvas, parent)
        mol_toolbar.config(background=BG_SECONDARY)
        for child in mol_toolbar.winfo_children():
            try:
                child.config(background=BG_SECONDARY)
            except Exception:
                pass
        mol_toolbar.update()

        self.mol_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Info label ────────────────────────────────────────────────────────
        self.mol_info_var = tk.StringVar(value="Enter a formula and click Build 3D")
        ttk.Label(parent, textvariable=self.mol_info_var,
                  style="Dim.TLabel").pack(pady=(2, 6))

        # ── Placeholder axes ──────────────────────────────────────────────────
        ax = self.mol_fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG_CARD)
        self.mol_fig.patch.set_facecolor(BG_CARD)
        ax.text(0, 0, 0,
                "Enter a formula or select a preset\nto visualise the 3D structure",
                ha="center", va="center", color=TEXT_DIM, fontsize=11, style="italic")
        ax.set_axis_off()
        self.mol_canvas.draw()

        # Cache last plotted molecule key so the label toggle can re-use it
        self._last_mol_key = None

    def _on_mol_preset_selected(self, event=None):
        """Load the molecule selected in the preset combobox and plot it."""
        val = self.mol_preset_var.get()
        if val.startswith("—"):
            return
        key = val.split("  —  ")[0].strip()
        print(f"[Molecule] Preset selected: {key!r}")
        self.mol_formula_var.set(key)
        self._plot_molecule_3d(key)

    def _refresh_molecule(self):
        """Re-plot the last drawn molecule (used by the label toggle)."""
        if self._last_mol_key:
            self._plot_molecule_3d(self._last_mol_key)

    def _build_molecule(self):
        """Resolve the formula/name typed in the entry field and plot it."""
        raw = self.mol_formula_var.get().strip()
        print(f"[Molecule] Build requested for: {raw!r}")
        if not raw:
            return

        # Direct key lookup (case-insensitive)
        key = next((k for k in MOLECULAR_DATABASE if k.lower() == raw.lower()), None)

        # Fallback: match by molecule name
        if key is None:
            key = next(
                (k for k, v in MOLECULAR_DATABASE.items()
                 if v["name"].lower() == raw.lower()),
                None,
            )

        if key is None:
            known = ", ".join(_MOL_KEYS_SORTED)
            messagebox.showinfo(
                "Not Found",
                f"'{raw}' is not in the built-in database.\n\n"
                f"Known formulas / names:\n{known}",
            )
            print(f"[Molecule] '{raw}' not found in database")
            return

        self._plot_molecule_3d(key)

    def _plot_molecule_3d(self, key):
        """Render the molecule identified by *key* in the 3-D axes.

        Atoms are drawn as scatter spheres coloured by element (CPK scheme).
        Bonds are drawn as grey line segments.  Element labels appear next to
        each atom when the Labels toggle is on.

        Drag  → rotate (handled by mpl_toolkits.mplot3d automatically)
        Scroll → zoom  (handled by matplotlib >= 3.3 natively)
        Toolbar → zoom-box, pan, save

        Parameters
        ----------
        key : str — key in MOLECULAR_DATABASE
        """
        print(f"[Molecule] Plotting: {key}")
        entry  = MOLECULAR_DATABASE[key]
        atoms  = entry["atoms"]   # list of (element, x, y, z)
        bonds  = _compute_bonds(atoms)
        show_labels = self.mol_labels_var.get()

        print(f"[Molecule]   atoms={len(atoms)}  bonds={len(bonds)}")

        self.mol_fig.clf()
        ax = self.mol_fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG_CARD)
        self.mol_fig.patch.set_facecolor(BG_CARD)

        # ── Draw bonds first (under atoms) ────────────────────────────────────
        for i, j in bonds:
            _, xi, yi, zi = atoms[i]
            _, xj, yj, zj = atoms[j]
            ax.plot(
                [xi, xj], [yi, yj], [zi, zj],
                color="#888888", linewidth=2.5, alpha=0.7, zorder=1,
            )

        # ── Draw atoms ────────────────────────────────────────────────────────
        seen_elems = {}
        for idx, (elem, x, y, z) in enumerate(atoms):
            props = ELEMENT_PROPS.get(elem, _ELEM_DEFAULT)
            color = props["color"]
            size  = props["size"]

            # First occurrence of each element → add to legend
            if elem not in seen_elems:
                seen_elems[elem] = color
                label = elem
            else:
                label = "_nolegend_"

            ax.scatter(
                [x], [y], [z],
                c=[color], s=size * 4,   # *4 because scatter s is in points²
                depthshade=True, edgecolors="#FFFFFF44",
                linewidths=0.5, label=label, zorder=2,
            )

            if show_labels:
                ax.text(x, y, z, f" {elem}", color=TEXT, fontsize=7,
                        ha="left", va="bottom", zorder=3)

        # ── Axis styling ──────────────────────────────────────────────────────
        ax.set_xlabel("x (Å)", color=TEXT_DIM, fontsize=8, labelpad=2)
        ax.set_ylabel("y (Å)", color=TEXT_DIM, fontsize=8, labelpad=2)
        ax.set_zlabel("z (Å)", color=TEXT_DIM, fontsize=8, labelpad=2)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.5)

        title = f"{entry['name']}  ({entry['formula']})   [{entry['category']}]"
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)

        # Legend (element colour key)
        leg = ax.legend(
            title="Elements", title_fontsize=8,
            facecolor=BG_SECONDARY, edgecolor=BORDER,
            fontsize=8, labelcolor=TEXT,
            loc="upper left", markerscale=0.6,
        )
        leg.get_title().set_color(TEXT_DIM)

        # Equal aspect ratio — centre the molecule
        coords = np.array([(x, y, z) for _, x, y, z in atoms])
        mid    = coords.mean(axis=0)
        rng    = max((coords.max(axis=0) - coords.min(axis=0)).max() / 2.0, 1.0)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)

        try:
            ax.set_box_aspect([1, 1, 1])   # matplotlib >= 3.3
        except AttributeError:
            pass

        self.mol_canvas.draw()
        self._last_mol_key = key

        info = (f"{entry['description']}   |   "
                f"{len(atoms)} atoms   {len(bonds)} bonds")
        self.mol_info_var.set(info)
        print(f"[Molecule] Done — {info}")

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

    # ══════════════════════════════════════════════════════════════════════════
    # Crystalline structure identification
    # ══════════════════════════════════════════════════════════════════════════

    def _identify_crystal_structure(self):
        """Identify the most likely crystalline / molecular structure(s) in the
        loaded Raman spectra by matching detected peaks against the built-in
        peak catalogue (RAMAN_MATERIALS).

        Algorithm
        ---------
        1. Compute mean spectrum from all loaded spectra.
        2. Normalise to [0, 1]; detect local maxima above threshold.
        3. Score each material in RAMAN_MATERIALS by weighted-recall
           (fraction of expected peak amplitude matched within ±TOLERANCE cm⁻¹).
        4. Display ranked candidates in the Results Summary panel.
        """
        print("[Crystal ID] Button clicked")

        try:
            if self.raman_spectra is None:
                print("[Crystal ID] No Raman data loaded — aborting")
                messagebox.showwarning(
                    "No Data",
                    "Load a Raman preset or CSV first, then click Identify Crystal Structure.",
                )
                return

            src = self.raman_processed if self.raman_processed is not None else self.raman_spectra
            wn  = self.raman_wavenumbers
            print(f"[Crystal ID] Data shape: {src.shape}, wn range: {wn[0]:.1f}–{wn[-1]:.1f} cm⁻¹")

            # ── Mean spectrum ──────────────────────────────────────────────────
            mean_spectrum = src.mean(axis=0)

            # ── Normalise to [0, 1] ───────────────────────────────────────────
            mn, mx = float(mean_spectrum.min()), float(mean_spectrum.max())
            print(f"[Crystal ID] Spectrum range: {mn:.4f} – {mx:.4f}")
            if mx == mn:
                messagebox.showwarning(
                    "Flat Spectrum",
                    "The mean spectrum has no variation — cannot detect peaks.\n"
                    "Check that data loaded correctly.",
                )
                return
            norm = (mean_spectrum - mn) / (mx - mn)

            # ── Tuning parameters ─────────────────────────────────────────────
            THRESHOLD   = 0.15   # Minimum normalised intensity
            TOLERANCE   = 25     # ±cm⁻¹ matching window
            MIN_DIST_CM = 20     # Minimum cm⁻¹ between peaks

            wn_step      = float(wn[1] - wn[0]) if len(wn) > 1 else 1.0
            min_dist_pts = max(1, int(MIN_DIST_CM / wn_step))
            print(f"[Crystal ID] wn_step={wn_step:.2f}, min_dist_pts={min_dist_pts}")

            # ── Peak detection ────────────────────────────────────────────────
            detected_peaks = []

            if _SCIPY_OK:
                print("[Crystal ID] Using scipy.signal.find_peaks")
                indices, _ = _find_peaks(norm, height=THRESHOLD, distance=min_dist_pts)
                for idx in indices:
                    detected_peaks.append((float(wn[idx]), float(norm[idx])))
            else:
                print("[Crystal ID] scipy not available — using numpy fallback")
                candidates = []
                for i in range(1, len(norm) - 1):
                    if norm[i] >= THRESHOLD and norm[i] > norm[i-1] and norm[i] > norm[i+1]:
                        candidates.append((float(wn[i]), float(norm[i])))
                candidates.sort(key=lambda x: -x[1])
                accepted = []
                for cand in candidates:
                    if all(abs(cand[0] - a[0]) >= MIN_DIST_CM for a in accepted):
                        accepted.append(cand)
                detected_peaks = sorted(accepted, key=lambda x: x[0])

            print(f"[Crystal ID] Detected {len(detected_peaks)} peaks: "
                  + ", ".join(f"{w:.0f}" for w, _ in detected_peaks[:10]))

            if not detected_peaks:
                messagebox.showinfo(
                    "No Peaks Detected",
                    f"No peaks above {THRESHOLD*100:.0f}% intensity threshold.\n"
                    "Try normalising the spectra or applying noise reduction first.",
                )
                return

            # ── Score each material (weighted recall) ─────────────────────────
            results = []
            for material, peaks in RAMAN_MATERIALS.items():
                total_weight   = sum(a for _, _, a in peaks)
                matched_weight = 0.0
                matched_count  = 0
                matched_peaks  = []

                for center, width, amp in peaks:
                    close = [(abs(dpk - center), dpk, di)
                             for dpk, di in detected_peaks
                             if abs(dpk - center) <= TOLERANCE]
                    if close:
                        close.sort()
                        _, dpk_wn, _ = close[0]
                        matched_weight += amp
                        matched_count  += 1
                        matched_peaks.append((center, dpk_wn, amp))

                score = matched_weight / total_weight if total_weight > 0 else 0.0
                results.append((material, score, matched_count, len(peaks), matched_peaks))
                print(f"[Crystal ID]   {material}: score={score*100:.1f}%  ({matched_count}/{len(peaks)} peaks)")

            results.sort(key=lambda x: -x[1])

            # ── Format results text ───────────────────────────────────────────
            lines = [
                "═══ CRYSTAL IDENTIFICATION ═══",
                f"Spectra averaged : {src.shape[0]}",
                f"Peaks detected   : {len(detected_peaks)}"
                f"  (threshold {THRESHOLD*100:.0f}%,  tol ±{TOLERANCE} cm⁻¹)",
                "",
                "── Top candidates ──────────────",
            ]

            BAR_W = 16
            for rank, (mat, score, matched, total, mpks) in enumerate(results[:5], start=1):
                filled = int(score * BAR_W)
                bar    = "█" * filled + "░" * (BAR_W - filled)
                conf   = "★★★" if score >= 0.80 else ("★★☆" if score >= 0.50 else "★☆☆")
                lines.append(f"\n#{rank}  {mat}")
                lines.append(f"    {bar}  {score*100:.1f}%  {conf}")
                lines.append(f"    Peaks matched: {matched}/{total}")
                if mpks:
                    detail = "  ".join(f"{ref:.0f}→{det:.0f}" for ref, det, _ in mpks[:4])
                    lines.append(f"    {detail}")

            lines += ["", "── Detected peaks ──────────────"]
            by_intensity = sorted(detected_peaks, key=lambda x: -x[1])
            for dpk_wn, dpk_int in by_intensity[:12]:
                lines.append(f"  {dpk_wn:7.1f} cm⁻¹   rel. intensity {dpk_int:.3f}")

            top_name, top_score = results[0][0], results[0][1]
            if top_score >= 0.50:
                lines.append(f"\n✓ Best match: {top_name} ({top_score*100:.1f}%)")
            else:
                lines.append("\n⚠ No confident match (top score < 50%)")
                lines.append("  Consider: mixed phase, coating, or unknown material.")

            print(f"[Crystal ID] Best match: {top_name} ({top_score*100:.1f}%)")
            self._update_stats("\n".join(lines))

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[Crystal ID] ERROR: {exc}\n{tb}")
            messagebox.showerror("Identification Error", f"{exc}\n\nSee terminal for details.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = PCAApp()    # Create and initialise the root window
    app.mainloop()    # Enter the Tk event loop (blocks until window is closed)
