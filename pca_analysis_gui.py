#!/usr/bin/env python3
"""
PCA Analysis GUI Application
─────────────────────────────
A desktop application for performing Principal Component Analysis
with interactive controls, preset datasets, and rich visualizations.

Libraries: tkinter, numpy, pandas, matplotlib, scikit-learn
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import io
import threading


# ─── Color Palette ───────────────────────────────────────────────────────────
BG = "#0f1117"
BG_SECONDARY = "#161b22"
BG_CARD = "#1c2333"
BORDER = "#2a3040"
TEXT = "#e6edf3"
TEXT_DIM = "#7d8590"
ACCENT = "#58a6ff"
ACCENT_HOVER = "#79c0ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"

PLOT_COLORS = ["#ff6b6b", "#51cf66", "#339af0", "#fcc419", "#cc5de8", "#22b8cf"]
PLOT_CMAPS = [
    ListedColormap(["#ffd43b", "#f8f9fa", "#96f2d7"]),
    ListedColormap(["#ffa8a8", "#d0ebff", "#b2f2bb"]),
]

# ─── Raman Spectroscopy Helpers ───────────────────────────────────────────────

RAMAN_WN = np.linspace(100, 3300, 650)  # shared wavenumber axis (cm⁻¹)

# Peak definitions: list of (center_cm⁻¹, half-width_cm⁻¹, relative_amplitude)
RAMAN_MATERIALS = {
    "Silicon":         [(521,  4,  1.00)],
    "Diamond":         [(1332, 6,  1.00)],
    "Graphene":        [(1350, 25, 0.35), (1580, 18, 1.00), (2700, 35, 0.65)],
    "TiO2 (Anatase)":  [(144,  6,  1.00), (197,  8,  0.15), (399,  12, 0.35),
                        (513,  10, 0.25), (639,  12, 0.45)],
    "Polystyrene":     [(621,  5,  0.25), (1001, 4,  1.00), (1031, 5,  0.40),
                        (1155, 5,  0.15), (1583, 8,  0.30), (1602, 6,  0.40),
                        (3054, 8,  0.55)],
    "Calcite":         [(280,  10, 0.30), (712,  8,  0.25), (1085, 7,  1.00)],
    "Quartz":          [(128,  8,  0.60), (206,  10, 0.50), (464,  10, 1.00),
                        (698,  8,  0.15), (795,  8,  0.20), (1082, 12, 0.15)],
    "PMMA":            [(813,  6,  0.45), (987,  6,  0.30), (1452, 8,  0.50),
                        (1727, 8,  1.00), (2953, 10, 0.70)],
    "Polyethylene":    [(1063, 8,  0.80), (1130, 8,  1.00), (1296, 6,  0.45),
                        (2848, 10, 0.90), (2883, 10, 1.00)],
    "Gypsum":          [(415,  8,  0.40), (492,  8,  0.55), (1008, 7,  1.00),
                        (1140, 6,  0.30), (3405, 12, 0.50)],
}

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


def _lorentzian(x, center, width, amp):
    return amp / (1.0 + ((x - center) / width) ** 2)


def _generate_raman_spectra(material, n_samples=20, noise=0.018, rng=None):
    """Synthetic Raman spectra with Lorentzian peaks, amplitude jitter, and noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    peaks = RAMAN_MATERIALS[material]
    spectra = []
    for _ in range(n_samples):
        s = np.zeros(len(RAMAN_WN))
        for center, width, amp in peaks:
            s += _lorentzian(RAMAN_WN, center + rng.uniform(-1.5, 1.5),
                             width, amp * rng.uniform(0.90, 1.10))
        # slight fluorescent background
        bg = rng.uniform(0, 0.04) * (RAMAN_WN - RAMAN_WN[0]) / (RAMAN_WN[-1] - RAMAN_WN[0])
        s += bg + rng.normal(0, noise, len(RAMAN_WN))
        spectra.append(np.clip(s, 0, None))
    return np.array(spectra)


class PCAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PCA Analysis Tool")
        self.geometry("1280x820")
        self.configure(bg=BG)
        self.minsize(1100, 700)

        self.data_X = None
        self.data_y = None
        self.feature_names = None
        self.target_names = None
        self.pca_result = None

        # Raman state
        self.raman_wavenumbers = None
        self.raman_spectra = None
        self.raman_labels = None
        self.raman_ax = None
        self._raman_vline = None

        self._build_styles()
        self._build_layout()

    # ── Styles ────────────────────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground=ACCENT)
        style.configure("Dim.TLabel", foreground=TEXT_DIM, font=("Segoe UI", 9))
        style.configure("Stat.TLabel", font=("Consolas", 11), foreground=CYAN)

        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#000",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 8),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_HOVER), ("disabled", BORDER)],
            foreground=[("disabled", TEXT_DIM)],
        )

        style.configure(
            "TButton",
            background=BG_SECONDARY,
            foreground=TEXT,
            font=("Segoe UI", 10),
            padding=(12, 6),
            borderwidth=1,
        )
        style.map("TButton", background=[("active", BG_CARD)])

        style.configure(
            "TCombobox", fieldbackground=BG_SECONDARY, background=BG_CARD,
            foreground=TEXT, selectbackground=ACCENT, selectforeground="#000",
        )
        style.map("TCombobox", fieldbackground=[("readonly", BG_SECONDARY)])

        style.configure(
            "TSpinbox", fieldbackground=BG_SECONDARY, background=BG_CARD,
            foreground=TEXT, arrowcolor=TEXT,
        )

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=BG_SECONDARY, foreground=TEXT_DIM,
            padding=(14, 6), font=("Segoe UI", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", BG_CARD)],
            foreground=[("selected", ACCENT)],
        )

        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG_SECONDARY)

    # ── Layout ────────────────────────────────────────────────────────────
    def _build_layout(self):
        # Title bar
        title_frame = ttk.Frame(self)
        title_frame.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(title_frame, text="◆  PCA Analysis Tool", font=("Segoe UI", 16, "bold"), foreground=ACCENT).pack(side="left")
        ttk.Label(title_frame, text="Principal Component Analysis with Python", style="Dim.TLabel").pack(side="left", padx=(12, 0))

        sep = tk.Frame(self, height=1, bg=BORDER)
        sep.pack(fill="x", padx=16, pady=(4, 8))

        # Main paned layout
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # ── Left Panel (controls) ──
        left = ttk.Frame(main, width=340)
        main.add(left, weight=0)
        self._build_controls(left)

        # ── Right Panel (results) ──
        right = ttk.Frame(main)
        main.add(right, weight=1)
        self._build_results(right)

    def _build_controls(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0, width=320)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=310)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        p = scroll_frame  # shorthand

        # ── Data Source ──
        self._section(p, "DATA SOURCE")

        ttk.Label(p, text="Preset Dataset").pack(anchor="w", padx=8, pady=(4, 2))
        self.preset_var = tk.StringVar(value="-- Select Preset --")
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
        presets.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(p, text="─ or ─", foreground=TEXT_DIM, font=("Segoe UI", 9)).pack(pady=4)

        ttk.Button(p, text="📂  Load CSV File", command=self._load_csv).pack(padx=8, fill="x")

        # Data info
        self.data_info_var = tk.StringVar(value="No data loaded")
        ttk.Label(p, textvariable=self.data_info_var, style="Dim.TLabel", wraplength=280).pack(padx=8, pady=(6, 2), anchor="w")

        # ── PCA Parameters ──
        self._section(p, "PCA PARAMETERS")

        ttk.Label(p, text="Number of Components").pack(anchor="w", padx=8, pady=(4, 2))
        self.n_components_var = tk.IntVar(value=2)
        spin = ttk.Spinbox(p, from_=1, to=50, textvariable=self.n_components_var, width=8)
        spin.pack(anchor="w", padx=8)

        ttk.Label(p, text="Test Split Ratio").pack(anchor="w", padx=8, pady=(8, 2))
        self.test_ratio_var = tk.DoubleVar(value=0.2)
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

        self.standardize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="Standardize features (recommended)", variable=self.standardize_var).pack(anchor="w", padx=8, pady=(6, 0))

        self.classify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="Run Logistic Regression classifier", variable=self.classify_var).pack(anchor="w", padx=8, pady=(2, 0))

        # ── Raman Viewer ──
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

        self.raman_normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(p, text="Normalize spectra (max = 1)",
                        variable=self.raman_normalize_var,
                        command=self._plot_raman_spectra).pack(anchor="w", padx=8, pady=(8, 0))

        self.raman_stack_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(p, text="Stack with vertical offset",
                        variable=self.raman_stack_var,
                        command=self._plot_raman_spectra).pack(anchor="w", padx=8, pady=(2, 4))

        # ── Run ──
        self._section(p, "")
        self.run_btn = ttk.Button(p, text="▶  Run PCA Analysis", style="Accent.TButton", command=self._run_analysis)
        self.run_btn.pack(padx=8, pady=4, fill="x")

        self.progress = ttk.Progressbar(p, mode="indeterminate", style="Horizontal.TProgressbar")
        self.progress.pack(padx=8, fill="x", pady=(0, 4))

        # ── Quick Stats ──
        self._section(p, "RESULTS SUMMARY")
        self.stats_text = tk.Text(
            p, height=14, bg=BG_SECONDARY, fg=CYAN,
            font=("Consolas", 9), relief="flat", bd=0,
            insertbackground=CYAN, selectbackground=ACCENT,
            wrap="word", padx=8, pady=6,
        )
        self.stats_text.pack(padx=8, fill="x", pady=(2, 8))
        self.stats_text.insert("1.0", "Run an analysis to see results...")
        self.stats_text.config(state="disabled")

    def _build_results(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        # Tab frames
        self.tab_raman = ttk.Frame(self.notebook)
        self.tab_scatter = ttk.Frame(self.notebook)
        self.tab_variance = ttk.Frame(self.notebook)
        self.tab_decision = ttk.Frame(self.notebook)
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_data = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_raman,    text="  Raman Spectra  ")
        self.notebook.add(self.tab_scatter,  text="  PCA Scatter  ")
        self.notebook.add(self.tab_variance, text="  Variance  ")
        self.notebook.add(self.tab_decision, text="  Decision Boundary  ")
        self.notebook.add(self.tab_heatmap,  text="  Component Heatmap  ")
        self.notebook.add(self.tab_data,     text="  Data Preview  ")

        self._build_raman_tab(self.tab_raman)

        # Figures dict
        self.figures = {}
        self.canvases = {}

        for name, tab in [
            ("scatter", self.tab_scatter),
            ("variance", self.tab_variance),
            ("decision", self.tab_decision),
            ("heatmap", self.tab_heatmap),
        ]:
            fig = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab)
            toolbar.config(background=BG_SECONDARY)
            for child in toolbar.winfo_children():
                try:
                    child.config(background=BG_SECONDARY)
                except Exception:
                    pass
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name] = fig
            self.canvases[name] = canvas

        # Data preview tab with text widget
        self.data_text = scrolledtext.ScrolledText(
            self.tab_data, bg=BG_SECONDARY, fg=TEXT,
            font=("Consolas", 9), relief="flat", bd=0,
            insertbackground=TEXT, selectbackground=ACCENT,
        )
        self.data_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Placeholder text on all plots
        for name, fig in self.figures.items():
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)
            ax.text(0.5, 0.5, "Load data and run analysis", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color=TEXT_DIM, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.canvases[name].draw()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _section(self, parent, title):
        if title:
            f = ttk.Frame(parent)
            f.pack(fill="x", padx=8, pady=(12, 2))
            ttk.Label(f, text=title, font=("Segoe UI", 9, "bold"), foreground=TEXT_DIM).pack(side="left")
            sep = tk.Frame(f, height=1, bg=BORDER)
            sep.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=1)

    def _style_ax(self, ax, title=""):
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        if title:
            ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)

    def _update_stats(self, text):
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

    # ── Data Loading ──────────────────────────────────────────────────────
    def _on_preset_selected(self, event=None):
        sel = self.preset_var.get()
        if "Iris" in sel:
            d = load_iris()
            self.data_X = pd.DataFrame(d.data, columns=d.feature_names)
            self.data_y = d.target
            self.feature_names = list(d.feature_names)
            self.target_names = list(d.target_names)
        elif "Wine" in sel:
            d = load_wine()
            self.data_X = pd.DataFrame(d.data, columns=d.feature_names)
            self.data_y = d.target
            self.feature_names = list(d.feature_names)
            self.target_names = list(d.target_names)
        elif "Breast Cancer" in sel:
            d = load_breast_cancer()
            self.data_X = pd.DataFrame(d.data, columns=d.feature_names)
            self.data_y = d.target
            self.feature_names = list(d.feature_names)
            self.target_names = list(d.target_names)
        elif "Blobs" in sel:
            from sklearn.datasets import make_blobs
            X, y = make_blobs(n_samples=400, n_features=5, centers=4, random_state=42)
            self.feature_names = [f"Feature_{i+1}" for i in range(5)]
            self.data_X = pd.DataFrame(X, columns=self.feature_names)
            self.data_y = y
            self.target_names = [f"Cluster {i}" for i in range(4)]
        elif "Circles" in sel:
            from sklearn.datasets import make_circles
            X, y = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=42)
            self.feature_names = ["X1", "X2"]
            self.data_X = pd.DataFrame(X, columns=self.feature_names)
            self.data_y = y
            self.target_names = ["Inner", "Outer"]
        else:
            return

        n_samples, n_features = self.data_X.shape
        n_classes = len(np.unique(self.data_y))
        self.data_info_var.set(
            f"✓ {sel.split('(')[0].strip()}: {n_samples} samples, "
            f"{n_features} features, {n_classes} classes"
        )
        self.n_components_var.set(min(2, n_features))
        self._show_data_preview()

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                messagebox.showerror("Error", "CSV must have at least 2 columns.")
                return

            # Assume last column is target
            self.feature_names = list(df.columns[:-1])
            self.data_X = df.iloc[:, :-1]
            self.data_y = df.iloc[:, -1].values

            # Encode target if strings
            if self.data_y.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.target_names = list(np.unique(self.data_y))
                self.data_y = le.fit_transform(self.data_y)
            else:
                self.target_names = [str(c) for c in np.unique(self.data_y)]

            n_samples, n_features = self.data_X.shape
            n_classes = len(np.unique(self.data_y))
            name = path.split("/")[-1].split("\\")[-1]
            self.data_info_var.set(
                f"✓ {name}: {n_samples} samples, "
                f"{n_features} features, {n_classes} classes"
            )
            self.n_components_var.set(min(2, n_features))
            self._show_data_preview()
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))

    def _show_data_preview(self):
        self.data_text.delete("1.0", "end")
        if self.data_X is not None:
            buf = io.StringIO()
            df = self.data_X.copy()
            df["target"] = self.data_y
            buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n")
            buf.write("=" * 60 + "\n\n")
            buf.write("First 20 rows:\n")
            buf.write(df.head(20).to_string(index=True))
            buf.write("\n\n" + "=" * 60 + "\n")
            buf.write("\nDescriptive Statistics:\n")
            buf.write(df.describe().to_string())
            self.data_text.insert("1.0", buf.getvalue())

    # ── Analysis ──────────────────────────────────────────────────────────
    def _run_analysis(self):
        if self.data_X is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return

        self.run_btn.config(state="disabled")
        self.progress.start(10)

        thread = threading.Thread(target=self._do_analysis, daemon=True)
        thread.start()

    def _do_analysis(self):
        try:
            X = self.data_X.values.astype(float)
            y = self.data_y
            n_comp = self.n_components_var.get()
            test_ratio = self.test_ratio_var.get()
            do_standardize = self.standardize_var.get()
            do_classify = self.classify_var.get()

            n_comp = min(n_comp, X.shape[1], X.shape[0])

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=0, stratify=y if len(np.unique(y)) > 1 else None
            )

            # Scale
            if do_standardize:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
            else:
                X_train_s = X_train
                X_test_s = X_test

            # PCA
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train_s)
            X_test_pca = pca.transform(X_test_s)

            explained = pca.explained_variance_ratio_
            cumulative = np.cumsum(explained)

            # Full PCA for scree plot
            pca_full = PCA(n_components=min(X.shape[1], X.shape[0]))
            if do_standardize:
                pca_full.fit(scaler.fit_transform(X))
            else:
                pca_full.fit(X)
            full_explained = pca_full.explained_variance_ratio_

            stats_lines = []
            stats_lines.append(f"Components: {n_comp}")
            stats_lines.append(f"Train/Test: {len(y_train)}/{len(y_test)}")
            stats_lines.append(f"─────────────────────────")
            for i, v in enumerate(explained):
                stats_lines.append(f"PC{i+1} variance: {v:.4f} ({v*100:.1f}%)")
            stats_lines.append(f"─────────────────────────")
            stats_lines.append(f"Total explained: {cumulative[-1]*100:.1f}%")

            classifier = None
            y_pred = None
            accuracy = None

            if do_classify and len(np.unique(y)) > 1:
                classifier = LogisticRegression(random_state=0, max_iter=1000)
                classifier.fit(X_train_pca, y_train)
                y_pred = classifier.predict(X_test_pca)
                accuracy = accuracy_score(y_test, y_pred)

                cm = confusion_matrix(y_test, y_pred)
                stats_lines.append(f"\nClassifier Accuracy: {accuracy*100:.1f}%")
                stats_lines.append(f"\nConfusion Matrix:")
                stats_lines.append(str(cm))

            self.pca_result = {
                "pca": pca,
                "X_train_pca": X_train_pca,
                "X_test_pca": X_test_pca,
                "y_train": y_train,
                "y_test": y_test,
                "explained": explained,
                "cumulative": cumulative,
                "full_explained": full_explained,
                "classifier": classifier,
                "y_pred": y_pred,
                "accuracy": accuracy,
            }

            self.after(0, lambda: self._update_stats("\n".join(stats_lines)))
            self.after(0, self._plot_all)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
        finally:
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.run_btn.config(state="normal"))

    # ── Plotting ──────────────────────────────────────────────────────────
    def _plot_all(self):
        self._plot_scatter()
        self._plot_variance()
        self._plot_decision()
        self._plot_heatmap()

    def _plot_scatter(self):
        r = self.pca_result
        fig = self.figures["scatter"]
        fig.clf()

        X_pca = r["X_train_pca"]
        y = r["y_train"]
        classes = np.unique(y)

        if X_pca.shape[1] >= 2:
            ax = fig.add_subplot(111)
            self._style_ax(ax, "PCA — First Two Components")
            for i, cls in enumerate(classes):
                mask = y == cls
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = self.target_names[cls] if cls < len(self.target_names) else f"Class {cls}"
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
                           alpha=0.75, s=40, edgecolors="white", linewidths=0.3)
            ax.set_xlabel("PC 1", fontsize=10)
            ax.set_ylabel("PC 2", fontsize=10)
            leg = ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER, fontsize=9, labelcolor=TEXT)
        else:
            ax = fig.add_subplot(111)
            self._style_ax(ax, "PCA — First Component")
            for i, cls in enumerate(classes):
                mask = y == cls
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = self.target_names[cls] if cls < len(self.target_names) else f"Class {cls}"
                ax.scatter(X_pca[mask, 0], np.zeros(mask.sum()), c=color, label=label,
                           alpha=0.75, s=40, edgecolors="white", linewidths=0.3)
            ax.set_xlabel("PC 1", fontsize=10)
            leg = ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER, fontsize=9, labelcolor=TEXT)

        self.canvases["scatter"].draw()

    def _plot_variance(self):
        r = self.pca_result
        fig = self.figures["variance"]
        fig.clf()

        full_exp = r["full_explained"]
        n = len(full_exp)
        x = np.arange(1, n + 1)

        ax1 = fig.add_subplot(121)
        self._style_ax(ax1, "Scree Plot")
        ax1.bar(x, full_exp * 100, color=ACCENT, alpha=0.8, edgecolor="none")
        ax1.plot(x, full_exp * 100, "o-", color=ORANGE, markersize=4, linewidth=1.5)
        ax1.set_xlabel("Component", fontsize=9)
        ax1.set_ylabel("Variance Explained (%)", fontsize=9)
        if n <= 20:
            ax1.set_xticks(x)

        ax2 = fig.add_subplot(122)
        self._style_ax(ax2, "Cumulative Variance")
        cum = np.cumsum(full_exp) * 100
        ax2.fill_between(x, cum, color=GREEN, alpha=0.2)
        ax2.plot(x, cum, "o-", color=GREEN, markersize=4, linewidth=2)
        ax2.axhline(y=95, color=RED, linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(n * 0.7, 96, "95% threshold", color=RED, fontsize=8)
        ax2.set_xlabel("Component", fontsize=9)
        ax2.set_ylabel("Cumulative Variance (%)", fontsize=9)
        ax2.set_ylim(0, 105)
        if n <= 20:
            ax2.set_xticks(x)

        fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96)
        self.canvases["variance"].draw()

    def _plot_decision(self):
        r = self.pca_result
        fig = self.figures["decision"]
        fig.clf()

        classifier = r["classifier"]
        if classifier is None or r["X_train_pca"].shape[1] < 2:
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)
            msg = "Enable classifier with ≥2 components\nto see decision boundaries"
            ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color=TEXT_DIM)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.canvases["decision"].draw()
            return

        X_train = r["X_train_pca"][:, :2]
        y_train = r["y_train"]
        X_test = r["X_test_pca"][:, :2]
        y_test = r["y_test"]
        classes = np.unique(y_train)

        # Retrain a 2D classifier for boundary visualization
        clf2d = LogisticRegression(random_state=0, max_iter=1000)
        clf2d.fit(X_train, y_train)

        for idx, (X_set, y_set, title_str) in enumerate([
            (X_train, y_train, "Decision Boundary (Train)"),
            (X_test, y_test, "Decision Boundary (Test)"),
        ]):
            ax = fig.add_subplot(1, 2, idx + 1)
            self._style_ax(ax, title_str)

            h = 0.05
            x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
            y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            cmap_bg = PLOT_CMAPS[idx % len(PLOT_CMAPS)]
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)

            for i, cls in enumerate(classes):
                mask = y_set == cls
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = self.target_names[cls] if cls < len(self.target_names) else f"Class {cls}"
                ax.scatter(X_set[mask, 0], X_set[mask, 1], c=color, label=label,
                           s=30, edgecolors="white", linewidths=0.3, alpha=0.85)

            ax.set_xlabel("PC 1", fontsize=9)
            ax.set_ylabel("PC 2", fontsize=9)
            leg = ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER, fontsize=8, labelcolor=TEXT, loc="best")

        fig.subplots_adjust(wspace=0.3, left=0.08, right=0.96)
        self.canvases["decision"].draw()

    def _plot_heatmap(self):
        r = self.pca_result
        fig = self.figures["heatmap"]
        fig.clf()

        pca = r["pca"]
        components = pca.components_
        n_comp = components.shape[0]
        n_feat = components.shape[1]

        ax = fig.add_subplot(111)
        self._style_ax(ax, "PCA Component Loadings")

        feat_labels = self.feature_names[:n_feat] if self.feature_names else [f"F{i}" for i in range(n_feat)]
        # Truncate long labels
        feat_labels = [l[:18] + "…" if len(l) > 18 else l for l in feat_labels]

        im = ax.imshow(components, cmap="RdBu_r", aspect="auto", interpolation="nearest")
        ax.set_yticks(range(n_comp))
        ax.set_yticklabels([f"PC{i+1}" for i in range(n_comp)], fontsize=9, color=TEXT)
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=7, color=TEXT_DIM)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
        cbar.outline.set_edgecolor(BORDER)

        # Annotate cells if not too many
        if n_comp * n_feat <= 120:
            for i in range(n_comp):
                for j in range(n_feat):
                    val = components[i, j]
                    color = "#000" if abs(val) > 0.4 else TEXT_DIM
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

        fig.subplots_adjust(bottom=0.22, left=0.1, right=0.95)
        self.canvases["heatmap"].draw()

    # ── Raman Tab ────────────────────────────────────────────────────────────
    def _build_raman_tab(self, parent):
        # Top control bar
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(ctrl, text="Interactive Raman Viewer", style="Header.TLabel").pack(side="left")

        # Figure + toolbar
        self.raman_fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_CARD)
        self.raman_fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10)
        self.raman_canvas = FigureCanvasTkAgg(self.raman_fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.raman_canvas, parent)
        toolbar.config(background=BG_SECONDARY)
        for child in toolbar.winfo_children():
            try:
                child.config(background=BG_SECONDARY)
            except Exception:
                pass
        toolbar.update()
        self.raman_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Cursor readout
        self.raman_cursor_var = tk.StringVar(
            value="Hover over the plot — wavenumber and intensity will appear here"
        )
        ttk.Label(parent, textvariable=self.raman_cursor_var,
                  style="Dim.TLabel").pack(pady=(2, 6))

        # Connect hover
        self.raman_canvas.mpl_connect("motion_notify_event", self._on_raman_hover)

        # Placeholder
        ax = self.raman_fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        ax.text(0.5, 0.5, "Select a Raman preset or load a CSV to view spectra",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color=TEXT_DIM, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.raman_canvas.draw()

    def _on_raman_preset_selected(self, event=None):
        sel = self.raman_preset_var.get()
        groups = RAMAN_PRESETS.get(sel)
        if not groups:
            return
        rng = np.random.default_rng(0)
        all_spectra, all_labels = [], []
        for material, n in groups:
            sp = _generate_raman_spectra(material, n_samples=n, rng=rng)
            all_spectra.append(sp)
            all_labels.extend([material] * n)
        self._set_raman_data(RAMAN_WN, np.vstack(all_spectra), all_labels)

    def _load_raman_csv(self):
        """Load Raman CSV.

        Format A (most common — e.g. Renishaw/WiTec export):
            First column  = wavenumber axis.
            Other columns = one spectrum each; column header = sample label.

        Format B (transposed — rows are spectra):
            First column  = sample label.
            Other columns = intensity at each wavenumber; column header = wavenumber.
        """
        path = filedialog.askopenfilename(
            title="Select Raman CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                messagebox.showerror("Error", "CSV must have at least 2 columns.")
                return

            first_header = str(df.columns[0]).lower().strip()
            wn_keywords = {"wavenumber", "wave", "raman shift", "cm-1", "cm⁻¹",
                           "shift", "wavenumbers", "raman_shift"}

            if first_header in wn_keywords:
                # Format A: first column is wavenumber
                wavenumbers = df.iloc[:, 0].values.astype(float)
                spectra = df.iloc[:, 1:].values.T.astype(float)
                labels = list(df.columns[1:])
            else:
                # Try numeric first column → Format A
                try:
                    wavenumbers = df.iloc[:, 0].values.astype(float)
                    spectra = df.iloc[:, 1:].values.T.astype(float)
                    labels = list(df.columns[1:])
                except (ValueError, TypeError):
                    # Format B: wavenumbers are column headers
                    wavenumbers = np.array(df.columns[1:], dtype=float)
                    spectra = df.iloc[:, 1:].values.astype(float)
                    labels = list(df.iloc[:, 0].astype(str))

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
        """Store Raman data and also wire it into the PCA pipeline."""
        self.raman_wavenumbers = wavenumbers
        self.raman_spectra = spectra
        self.raman_labels = labels

        unique_labels = sorted(set(labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}

        col_names = [f"{w:.1f}" for w in wavenumbers]
        self.data_X = pd.DataFrame(spectra, columns=col_names)
        self.data_y = np.array([label_to_int[l] for l in labels])
        self.feature_names = col_names
        self.target_names = unique_labels

        n_samples, n_features = self.data_X.shape
        n_classes = len(unique_labels)
        self.data_info_var.set(
            f"✓ Raman: {n_samples} spectra, {n_features} wavenumbers, {n_classes} classes"
        )
        self.n_components_var.set(min(3, n_samples - 1, n_features))
        self._show_data_preview()
        self._plot_raman_spectra()
        self.notebook.select(self.tab_raman)

    def _plot_raman_spectra(self):
        if self.raman_spectra is None:
            return

        fig = self.raman_fig
        fig.clf()
        ax = fig.add_subplot(111)
        self._style_ax(ax, "Raman Spectra")

        wn = self.raman_wavenumbers
        spectra = self.raman_spectra.copy()
        labels = self.raman_labels
        unique_labels = sorted(set(labels))

        if self.raman_normalize_var.get():
            mx = spectra.max(axis=1, keepdims=True)
            mx[mx == 0] = 1
            spectra = spectra / mx

        stack = self.raman_stack_var.get()
        offset_step = spectra.max() * 0.18 if stack else 0.0

        seen = {}
        for i, (spectrum, label) in enumerate(zip(spectra, labels)):
            color = PLOT_COLORS[unique_labels.index(label) % len(PLOT_COLORS)]
            y = spectrum + offset_step * unique_labels.index(label)
            lbl = label if label not in seen else "_nolegend_"
            line, = ax.plot(wn, y, color=color, alpha=0.65, linewidth=0.9, label=lbl)
            if label not in seen:
                seen[label] = line

        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.legend(facecolor=BG_SECONDARY, edgecolor=BORDER,
                  fontsize=9, labelcolor=TEXT)

        # Persistent vertical cursor line (invisible until hover)
        self._raman_vline = ax.axvline(x=wn[0], color=ACCENT, linewidth=1.0,
                                        linestyle="--", alpha=0.0)
        self.raman_ax = ax
        self.raman_canvas.draw()

    def _on_raman_hover(self, event):
        if self.raman_ax is None or event.inaxes != self.raman_ax:
            if self._raman_vline is not None and self._raman_vline.get_alpha() > 0:
                self._raman_vline.set_alpha(0.0)
                self.raman_canvas.draw_idle()
            return
        x, y = event.xdata, event.ydata
        if x is None:
            return
        self.raman_cursor_var.set(
            f"Wavenumber: {x:,.1f} cm⁻¹   |   Intensity: {y:.4f}"
        )
        if self._raman_vline is not None:
            self._raman_vline.set_xdata([x, x])
            self._raman_vline.set_alpha(0.7)
            self.raman_canvas.draw_idle()


if __name__ == "__main__":
    app = PCAApp()
    app.mainloop()
