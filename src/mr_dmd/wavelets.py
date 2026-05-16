"""
wavelet_bins.py
---------------
Dyadic multi-resolution wavelet windows with arbitrary-point evaluation.

At layer l there are  N = 2^(l-1)  equal-width bins covering [t_start, t_end].
Each bin is a WaveletWindow object: callable at any time point t, including
arrays.  Outside [start, stop] it returns 0 by default.

Key fix: the wavelet's *effective* support (where amplitude is non-negligible)
is mapped onto [start, stop], so the window fills the interval fully rather
than being flat for most of it.
"""

import numpy as np
import pywt
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Helper: effective (non-flat) support of a wavelet
# ---------------------------------------------------------------------------


def effective_support(wavelet_name: str, threshold: float = 0.01) -> tuple[float, float]:
    """
    Return the x-range where |psi(x)| >= threshold * max|psi|.

    Many wavelets (e.g. mexh) have a declared natural support much wider than
    where the function is non-negligible.  Using the effective support ensures
    the wavelet fills [start, stop] without flat tails at the boundaries.

    Parameters
    ----------
    wavelet_name : str
    threshold : float
        Fraction of peak amplitude below which values are treated as flat.
        0.01 (1% of peak) is a good default; use 0.001 for a looser crop.

    Returns
    -------
    (x_lo, x_hi) on the wavelet's natural coordinate axis.
    """
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    psi, x = wavelet.wavefun(level=10)
    above = np.where(np.abs(psi) >= threshold * np.max(np.abs(psi)))[0]
    if len(above) == 0:
        return float(x[0]), float(x[-1])
    return float(x[above[0]]), float(x[above[-1]])


# ---------------------------------------------------------------------------
# WaveletWindow — callable, interpolated wavelet on [start, stop]
# ---------------------------------------------------------------------------


class WaveletWindow:
    """
    A wavelet scaled to fill [start, stop], evaluable at any t.

    The wavelet's *effective* support is mapped onto [start, stop], so the
    window rises and falls near the interval boundaries rather than being
    flat for most of the bin.

    Construction pre-computes a CubicSpline, so repeated calls are cheap.

    Parameters
    ----------
    wavelet_name : str
        Any pywt continuous wavelet, e.g. 'mexh', 'morl', 'gaus2'.
    start, stop : float
        Interval bounds.
    num_points : int
        Internal grid density for the spline.
    extrapolate : bool
        If False (default), evaluations outside [start, stop] return 0.
    threshold : float
        Controls how aggressively flat tails are cropped (default 0.01 = 1%).
        Lower values (e.g. 0.001) give a slightly wider effective support.

    Usage
    -----
    >>> w = WaveletWindow('mexh', 2.0, 5.0)
    >>> w(3.5)                      # scalar -> float
    >>> w(np.linspace(0, 8, 1000))  # array  -> np.ndarray, 0 outside [2,5]
    """

    def __init__(
        self,
        wavelet_name: str,
        start: float,
        stop: float,
        num_points: int = 512,
        extrapolate: bool = False,
        threshold: float = 0.01,
    ):
        self.wavelet_name = wavelet_name
        self.start = start
        self.stop = stop
        self.extrapolate = extrapolate

        wavelet = pywt.ContinuousWavelet(wavelet_name)
        psi, x_nat = wavelet.wavefun(level=10)

        # Map effective support (not full declared support) onto [start, stop]
        x_lo, x_hi = effective_support(wavelet_name, threshold=threshold)

        t_grid = np.linspace(start, stop, num_points)
        x_query = x_lo + (t_grid - start) / (stop - start) * (x_hi - x_lo)
        w_grid = np.interp(x_query, x_nat, np.real(psi))

        # Cubic spline: C2 continuity, O(log n) per evaluation
        self._spline = CubicSpline(t_grid, w_grid, extrapolate=extrapolate)

    def __call__(self, t):
        """
        Evaluate the wavelet window at time(s) t.

        Returns 0 outside [start, stop] unless extrapolate=True.
        """
        scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        values = self._spline(t_arr)

        if not self.extrapolate:
            values[(t_arr < self.start) | (t_arr > self.stop)] = 0.0
        return float(values[0]) if scalar else values

    def __repr__(self):
        return f"WaveletWindow('{self.wavelet_name}', " f"start={self.start}, stop={self.stop})"


# ---------------------------------------------------------------------------
# Dyadic layer: 2^(l-1) WaveletWindow objects covering [t_start, t_end]
# ---------------------------------------------------------------------------


def dyadic_wavelet_layer(
    wavelet_name: str,
    t_start: float,
    t_end: float,
    layer: int,
    n_funcs: int = 1,
    num_points_per_bin: int = 512,
    extrapolate=False,
    threshold: float = 0.01,
) -> list:
    """
    Return WaveletWindow objects for all bins at layer l.

    Layer l splits [t_start, t_end] into 2^(l-1) equal bins.
    """
    n_bins = 2 ** (layer - 1)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    l = n_funcs - 1

    return [
        WaveletWindow(
            wavelet_name,
            bin_edges[k] + n * (bin_edges[k + 1] - bin_edges[k]) / l,
            bin_edges[k + 1] + n * (bin_edges[k + 1] - bin_edges[k]) / l,
            num_points=num_points_per_bin,
            extrapolate=extrapolate,
            threshold=threshold,
        )
        for k in range(n_bins)
        for n in range(-n_funcs // 2, n_funcs // 2 + 1)
    ]


# ---------------------------------------------------------------------------
# Multi-layer decomposition
# ---------------------------------------------------------------------------


def dyadic_wavelet_decomposition(
    wavelet_name: str,
    t_start: float,
    t_end: float,
    max_layer: int,
    n_funcs: int = 1,
    num_points_per_bin: int = 512,
    extrapolate=False,
    threshold: float = 0.01,
) -> dict:
    """
    Build WaveletWindow objects for layers 1 to max_layer.

    Returns dict: layer -> list[WaveletWindow]

    Example
    -------
    >>> decomp = dyadic_wavelet_decomposition('mexh', 0, 8, max_layer=4)
    >>> decomp[3][1](4.5)                      # scalar
    >>> decomp[3][1](np.linspace(0, 8, 200))   # array
    """
    return {
        l: dyadic_wavelet_layer(
            wavelet_name,
            t_start,
            t_end,
            l,
            n_funcs=n_funcs,
            num_points_per_bin=num_points_per_bin,
            extrapolate=extrapolate,
            threshold=threshold,
        )
        for l in range(1, max_layer + 1)
    }


# ---------------------------------------------------------------------------
# Demo / plot
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    WAVELET = "mexh"
    T_START = 0.0
    T_END = 8.0
    MAX_LAYER = 4
    N_FUNCS = 51

    decomp = dyadic_wavelet_decomposition(
        WAVELET, T_START, T_END, MAX_LAYER, n_funcs=N_FUNCS, extrapolate=False, threshold=0.01
    )
    t_query = np.linspace(T_START, T_END, 2000)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"Dyadic WaveletWindow  |  wavelet={WAVELET}  |  [{T_START}, {T_END}]",
        fontsize=13,
    )
    gs = gridspec.GridSpec(MAX_LAYER, 1, hspace=0.55)
    colors = plt.cm.tab10.colors

    for l, windows in decomp.items():
        ax = fig.add_subplot(gs[l - 1])
        n_bins = len(windows)
        ax.set_title(f"Layer {l}  -  {n_bins} bin{'s' if n_bins > 1 else ''}", fontsize=10)
        ax.set_xlim(T_START, T_END)
        for k, win in enumerate(windows):
            ax.plot(t_query, np.abs(win(t_query)), color=colors[k % len(colors)], lw=1.5)
            ax.axvline(win.start, color="grey", lw=0.5, ls="--")
        ax.axvline(T_END, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("amplitude", fontsize=8)

    ax.set_xlabel("time")

    # Spot-check
    w = decomp[1][0]
    test_pts = [0.0, w.start, (w.start + w.stop) / 2, w.stop, T_END]
    print(f"\nSpot-check  {w}")
    for tp in test_pts:
        print(f"  w({tp:.3f}) = {w(tp):.6f}")

    plt.savefig("dyadic_wavelet_windows.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nDone.")
