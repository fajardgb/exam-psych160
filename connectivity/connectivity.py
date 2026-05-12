# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.10.9",
#     "nilearn==0.11.1",
#     "numpy>=2.4.4",
#     "pandas>=3.0.2",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import os
    import time
    import sys

    import marimo as mo

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
    from nilearn.interfaces.fmriprep import load_confounds
    from nilearn.image import load_img, smooth_img, math_img
    from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
    from nilearn.plotting import plot_design_matrix, plot_roi, view_img, plot_stat_map

    import nilearn
    print(f"nilearn version: {nilearn.__version__}")
    return (
        FirstLevelModel,
        NiftiMasker,
        load_confounds,
        load_img,
        math_img,
        mo,
        np,
        os,
        pd,
        plot_design_matrix,
        plot_roi,
        plot_stat_map,
        plt,
        smooth_img,
        sys,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up directories and hyperparameters
    """)
    return


@app.cell
def _():
    # Hyperparameters
    TR = 2.0
    FWHM = 6.0
    N_SCANS = 182

    SEED_LABEL = 37

    num_runs = 3
    runs = [str(i + 1).zfill(2) for i in range(num_runs)]
    return FWHM, N_SCANS, SEED_LABEL, TR, runs


@app.cell
def _(FWHM, TR, os, runs, sys, time):
    # directories
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")

    # on HPC or on local machine?

    # if HPC, get sub ID from command line args
    if "dartfs" in current_dir:
        print("\nRunning on HPC")
        sub = sys.argv[1]
        sub = sub.zfill(2)  # zero-pad to 2 digits if needed

        data_dir = f"/dartfs/rc/lab/D/DBIC/DBIC/psych160/data/stopsignal/derivatives/fmriprep/sub-{sub}/func"
        mask_dir = f"/dartfs/rc/lab/D/DBIC/DBIC/psych160/masks/k50_2mm.nii.gz"

        # sleep for 10s if not sub-01 - for creating save dirs on HPC
        if sub != "01":
            time.sleep(10)  

    # if local, manually set sub ID here
    else:
        print("\nRunning locally")
        sub = "01"

        data_dir = f"../data/fmriprep/sub-{sub}/func"
        mask_dir = f"../data/masks/k50_2mm.nii.gz"

    print(f"\nRunning sub-{sub}")
    print(f"\nHyperparameters: TR={TR}, FWHM={FWHM}, runs={runs}")
    return current_dir, data_dir, mask_dir, sub


@app.cell
def _(current_dir, os, sub):
    # set and create save directories
    save_dir = f"{current_dir}/output/sub-{sub}"
    os.makedirs(save_dir, exist_ok=True)
    return (save_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the fMRI data for this sub
    """)
    return


@app.cell
def _(FWHM, data_dir, load_confounds, load_img, pd, runs, smooth_img, sub):
    fmri_imgs = []
    events = []
    confounds = []

    for run in runs:

        if sub == "12" and run == "03":
            print(f"Skipping sub-{sub} run-{run} due to missing run.")
            continue

        events_df = pd.read_csv(f"{data_dir}/sub-{sub}_task-stopsignal_run-{run}_events.tsv", sep="\t")
        events_df = events_df.dropna(subset=['duration']) # remove 1st col, has duration = NaN
        events_df['trial_type'] = events_df['trial_type'].str.replace(' ', '_') # Nilearn requires a python string format (no spaces)
        events_df = events_df[["onset", "duration", "trial_type"]] # keep only needed cols
        events.append(events_df)

        fmri_img_path = f"{data_dir}/sub-{sub}_task-stopsignal_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        _fmri_img = load_img(fmri_img_path)
        _fmri_img = smooth_img(_fmri_img, fwhm=FWHM) # apply smoothing
        fmri_imgs.append(_fmri_img)

        confound, _ = load_confounds(fmri_img_path, 
                                     strategy=["motion", "wm_csf", "compcor", "high_pass"],
                                     motion="full",
                                     wm_csf="basic")
        confounds.append(confound)

    assert len(events) == len(fmri_imgs) == len(confounds), f"Mismatch in events ({len(events)}) or fMRI images ({len(fmri_imgs)}) or confounds ({len(confounds)})."
    print(f"Loaded {len(events)} runs of data for sub-{sub}.")
    return confounds, fmri_imgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load parcel mask
    """)
    return


@app.cell
def _(current_dir, load_img, mask_dir, plot_roi, plt):
    parcel_mask = load_img(mask_dir)

    if "dartfs" not in current_dir:
        plot_roi(parcel_mask, title="k50 parcel mask", colorbar=True)
        plt.show()
    return (parcel_mask,)


@app.cell
def _(fmri_imgs, np, parcel_mask):
    # confirm parcel mask has same shape and affine as fMRI data
    assert parcel_mask.shape == fmri_imgs[0].shape[:3], f"Parcel mask shape {parcel_mask.shape} does not match fMRI image shape {fmri_imgs[0].shape}."
    assert np.allclose(parcel_mask.affine, fmri_imgs[0].affine), "Parcel mask affine does not match fMRI image affine."
    print("Parcel mask shape and affine match fMRI images.")
    return


@app.cell
def _(SEED_LABEL, current_dir, math_img, parcel_mask, plot_roi, plt):
    # create binary mask for seed region
    # need to add 1 to SEED_LABEL because parcel labels are 1-indexed, but SEED_LABEL is 0-indexed
    mask_img = math_img(
                f"np.where(np.round(img) == {SEED_LABEL + 1}, 1, 0)", img=parcel_mask
            )
    if "dartfs" not in current_dir:
        plot_roi(mask_img, title=f"Seed{SEED_LABEL} mask", colorbar=True)
        plt.savefig(f"seed{SEED_LABEL}_mask.png")
        plt.show()
    return (mask_img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract anterior insula (seed37) time series per run and create design matrix
    """)
    return


@app.cell
def _(np, pd):
    # function to find spikes, modified from nltools 
    def find_spikes(fmri_img, global_spike_cutoff=3, diff_spike_cutoff=3):
        """ 
        Function to identify global and differential spikes from fMRI Time Series Data

        Parameters:
            fmri_img: 4D Nifti image of fMRI data (X x Y x Z x T)
            global_spike_cutoff: cutoff to identify spikes in global signal in standard deviations
            diff_spike_cutoff: cutoff to identify spikes in average frame difference in standard deviations

        Returns:
            spikes_df: pandas dataframe with spikes as indicator variables
        """

        data = fmri_img.get_fdata()

        global_mn = np.mean(data, axis=(0,1,2))
        frame_diff = np.mean(np.abs(np.diff(data, axis=3)), axis=(0, 1, 2))

        global_outliers = np.append(
                np.where(
                    global_mn > np.mean(global_mn) + np.std(global_mn) * global_spike_cutoff
                ),
                np.where(
                    global_mn < np.mean(global_mn) - np.std(global_mn) * global_spike_cutoff
                ),
            )

        frame_outliers = np.append(
                np.where(
                    frame_diff
                    > np.mean(frame_diff) + np.std(frame_diff) * diff_spike_cutoff
                ),
                np.where(
                    frame_diff
                    < np.mean(frame_diff) - np.std(frame_diff) * diff_spike_cutoff
                ),
            )

        # build spike regressors
        spikes_df = pd.DataFrame(index=range(len(global_mn)))
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
def _(
    N_SCANS,
    NiftiMasker,
    SEED_LABEL,
    TR,
    confounds,
    find_spikes,
    fmri_imgs,
    mask_img,
    np,
    pd,
):
    smoothed_imgs = []
    design_matrices = []

    for i, (fmri_img, confound_df) in enumerate(zip(fmri_imgs, confounds)):

        # 1. Extract seed time series
        seed_masker = NiftiMasker(mask_img=mask_img, 
                                        standardize='zscore_sample', # z-score
                                        detrend=False, low_pass=None, high_pass=None, # this will be done in the GLM
                                        t_r=TR)
        seed_ts = seed_masker.fit_transform(fmri_img)
        seed_ts = seed_ts.mean(axis=1)
        assert seed_ts.shape[0] == confound_df.shape[0], f"Seed time series length {seed_ts.shape[0]} does not match number of confound time points {confound_df.shape[0]}."
        assert seed_ts.shape[0] == N_SCANS, f"Seed time series length {seed_ts.shape[0]} does not match expected number of scans {N_SCANS}."

        # 2. Create design matrix for this run
        seed_series = pd.Series(seed_ts, name=f"seed{SEED_LABEL}", index=confound_df.index)
        poly_df = pd.DataFrame({
        "trend_lin":  np.linspace(-1, 1, N_SCANS),
        "trend_quad": np.linspace(-1, 1, N_SCANS) ** 2,
        })
        spikes_df = find_spikes(fmri_img)
        run_dm = pd.concat([seed_series, confound_df, poly_df, spikes_df], axis=1)
        design_matrices.append(run_dm)
    return design_matrices, seed_ts


@app.cell
def _(SEED_LABEL, current_dir, plt, seed_ts):
    # plot an example time series
    if "dartfs" not in current_dir:
        plt.plot(seed_ts)
        plt.xlabel("Time (TR)")
        plt.ylabel("Mean Signal")
        plt.title(f"Seed{SEED_LABEL} Example Time Series")  
        plt.savefig(f"seed{SEED_LABEL}_timeseries.png")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Whole-brain GLM
    """)
    return


@app.cell
def _(FirstLevelModel, TR, design_matrices, fmri_imgs):
    # these parameters are included in the design matrix already!
    glm = FirstLevelModel(t_r=TR, 
                          hrf_model=None,
                          drift_model=None,
                          high_pass=None,
                          standardize=False, 
                          noise_model='ar1',
                          minimize_memory=True
                        )
    glm.fit(fmri_imgs, design_matrices=design_matrices)
    return (glm,)


@app.cell
def _(current_dir, glm, plot_design_matrix, plt, sub):
    if "dartfs" not in current_dir:
        dms = glm.design_matrices_

        fig, axes = plt.subplots(1, len(dms),
                                    figsize=(7 * len(dms), 8))
        if len(dms) == 1:
            axes = [axes]

        for _i, dm in enumerate(dms):
            # print(dm.columns)
            plot_design_matrix(dm, axes=axes[_i])
            axes[_i].set_title(f"sub-{sub} run-{_i+1} design matrix", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"design_matrices.png")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get betas (connectivity map)
    """)
    return


@app.cell
def _(SEED_LABEL, glm, save_dir, sub):
    seed_con_img = glm.compute_contrast(f"seed{SEED_LABEL}", output_type="effect_size")
    seed_con_img.to_filename(f"{save_dir}/sub-{sub}_seed{SEED_LABEL}_connectivity_beta.nii.gz")
    print(f"Saved seed connectivity map to {save_dir}/sub-{sub}_seed{SEED_LABEL}_connectivity_beta.nii.gz")
    return (seed_con_img,)


@app.cell
def _(SEED_LABEL, current_dir, plot_stat_map, plt, seed_con_img, sub):
    if "dartfs" not in current_dir:
        plot_stat_map(seed_con_img, title=f"sub-{sub} seed{SEED_LABEL} connectivity map (beta values)", threshold=1.5) # low threshold, since 1 sub betas
        plt.show()
    return


if __name__ == "__main__":
    app.run()
