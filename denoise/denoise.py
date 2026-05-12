# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dartbrains-tools>=0.1.3",
#     "marimo>=0.23.3",
#     "matplotlib>=3.10.9",
#     "nibabel>=5.4.2",
#     "numpy>=2.0",
#     "scikit-learn>=1.8.0",
#     "scipy>=1.13",
#     "seaborn>=0.13.2",
#     "statsmodels",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nibabel as nib
    import pandas as pd

    from numpy.fft import fft, ifft, fftfreq
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import butter, filtfilt
    # from nltools.stats import regress
    # from nltools.data import Design_Matrix
    from dartbrains_tools.notebook_utils import plot_timeseries

    from sklearn.decomposition import FastICA

    import statsmodels.api as sm

    return mo, np, pd, plot_timeseries, plt, sm, sns


@app.cell
def _(np, pd):
    # function to find spikes copied from nltools 
    def find_spikes(data, global_spike_cutoff=3, diff_spike_cutoff=3):
        """ 
        Function to identify spikes from a 1D timeseries.

        Parameters:
            data: 1D timeseries data
            global_spike_cutoff: cutoff to identify spikes in global signal in standard deviations
            diff_spike_cutoff: cutoff to identify spikes in average frame difference in standard deviations
            global_spike_cutoff: cutoff to identify spikes in global signal in standard deviations
            diff_spike_cutoff: cutoff to identify spikes in average frame difference in standard deviations

        Returns:
            spikes_df: pandas dataframe with spikes as indicator variables
        """

        data = np.array(data).squeeze()
        frame_diff = np.abs(np.diff(data))

        if global_spike_cutoff is not None:
            global_outliers = np.append(
                np.where(data > np.mean(data) + np.std(data) * global_spike_cutoff),
                np.where(data < np.mean(data) - np.std(data) * global_spike_cutoff),
            )

        if diff_spike_cutoff is not None:
            frame_outliers = np.append(
                np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * diff_spike_cutoff),
                np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * diff_spike_cutoff),
            )

        # build spike regressors
        spikes_df = pd.DataFrame(index=range(len(data)))
        if global_spike_cutoff is not None:
            for i, loc in enumerate(global_outliers):
                col = f"global_spike{str(i + 1)}"
                spikes_df[col] = 0
                spikes_df.loc[int(loc), col] = 1
        # build FD regressors
        if diff_spike_cutoff is not None:
            for i, loc in enumerate(frame_outliers):
                col = f"diff_spike{str(i + 1)}"
                spikes_df[col] = 0
                spikes_df.loc[int(loc), col] = 1
        return spikes_df

    return (find_spikes,)


@app.cell
def _(pd, plot_timeseries):
    TR = 2.0
    hp_cutoff = 0.01
    poly_order = 2

    data = pd.read_csv('Noisy_Signal.csv')
    original = data.copy()
    signal = data.values.squeeze()
    n = len(signal)
    print(f"Original signal has {n} time points.")

    plot_timeseries(data)
    return TR, data, n, original, poly_order, signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get spikes, cosine, poly, and intercept
    """)
    return


@app.cell
def _(TR, data, find_spikes, n, np, poly_order):
    # find global and local spikes
    spikes = find_spikes(data).values
    print(f"Identified {spikes.shape[1]} spikes in the data.")

    # DCT high-pass (nltools/dartbrains style: spm_dctmtx-based, unit-scaled)
    # duration is the filter length in seconds (e.g. 128s), analogous to add_dct_basis(duration=128)
    hp_duration = 180  # in seconds
    n_dct = int(np.fix(2 * (n * TR) / hp_duration))
    timepoints = np.arange(n)
    k          = np.arange(1, n_dct + 1)
    dct        = np.cos(
        np.pi * k[np.newaxis, :] * timepoints[:, np.newaxis] / n
    )
    # unit-scale each column to [-1, 1] (nltools unit_scale=True default)
    dct        = dct / np.abs(dct).max(axis=0, keepdims=True)
    print(f"Added {dct.shape[1]} DCT regressors for high-pass filtering.")

    # polynomial trend regressors
    t    = np.linspace(-1, 1, n)
    poly = np.column_stack([t ** i for i in range(1, poly_order + 1)])  # shape (n, poly_order)

    # intercept
    intercept = np.ones((n,1))

    # Combine everything
    X = np.hstack([spikes, dct, poly, intercept]) 
    return (X,)


@app.cell
def _(X, plt, sns):
    # plot design matrix
    plt.figure(figsize=(12, 6))
    sns.heatmap(X, cmap='viridis', cbar=False)
    plt.title('Design Matrix')
    plt.xlabel('Regressors')
    plt.ylabel('Time Points')
    plt.savefig('design_matrix.png', dpi=300)
    plt.show()
    return


@app.cell
def _(X, signal, sm):
    model = sm.OLS(signal, X).fit()

    print(model.summary())
    betas = model.params
    return (betas,)


@app.cell
def _(X, betas, signal):
    signal_clean  = signal - X @ betas   # residuals (nuisance removed, mean also removed)
    return (signal_clean,)


@app.cell
def _(pd, signal_clean):
    # write out signal_clean to csv
    pd.DataFrame(signal_clean, columns=['Denoised_Signal']).to_csv('Denoised_Signal.csv', index=False)
    return


@app.cell
def _(original, plt, signal_clean):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(original.values, color='steelblue', lw=0.8)
    axes[0].set_title('Original signal')
    axes[1].plot(signal_clean, color='darkorange', lw=0.8)
    axes[1].set_title('Denoised signal (spikes + DCT + polynomial regressed out)')
    axes[1].set_xlabel('Timepoint')
    fig.tight_layout()
    plt.savefig('denoised_signal.png', dpi=150)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Writeup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I used global and local spikes, discrete cosine transforms, and polynomial drift nuisance variables to regress out the noise in the signal. The denoised signal can be found in `Denoised_Signal.csv`.

    From the above plot above, we can clearly see that spikes were removed. We see the clear downwards trend removed as well.
    """)
    return


if __name__ == "__main__":
    app.run()
