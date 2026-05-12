# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "dartbrains-tools>=0.1.3",
#     "nilearn>=0.13,<0.14",
#     "nltools @ git+https://github.com/cosanlab/nltools.git@fix/numpy-2-compat",
#     "nibabel>=5.4",
#     "pybids>=0.22",
#     "numpy>=2.0",
#     "pandas>=2.2",
#     "scipy>=1.13",
#     "matplotlib>=3.9",
#     "seaborn>=0.13",
#     "tqdm>=4.66",
# ]
# ///
# Pin nilearn <0.14: 0.13.1 is the last release whose dataset fetchers can
# still download from the web (e.g. fetch_neurovault_motor_task,
# fetch_atlas_*); the 0.14 line removes those URLs.
#
# nltools from the fix/numpy-2-compat branch: the released nltools 0.5.1
# declares numpy<1.24, which conflicts with the rest of the modern stack
# (dartbrains-tools, nibabel>=5.4, etc., all require numpy>=1.25 or >=2.0).
# That branch drops the cap. Switch to `nltools>=0.5.2` once 0.5.2 ships.

import marimo

__generated_with = "0.23.5"
app = marimo.App(layout_file="layouts/Exam_Spring2026.grid.json")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exam Spring 2026

    Throughout the class we have been analyzing a simple dataset (e.g., Pinel Localizer). The goal of this exam is to analyze a *new* dataset that you have not worked on before. This is real dataset that has already been [published](https://www.jneurosci.org/content/26/9/2424) and is a classic paper exploring the neural basis of cognitive control. A PDF of the paper is included in the exam folder in case you would like to read more about the details of the task or experimental design.

    In this [study](https://www.jneurosci.org/content/26/9/2424), 13 adult participants completed a stop-signal task, in which they were instructed to occasionally inhibit a prepotent motor response upon hearing a tone. The difficulty of the task was titrated to each participant so that they were able to succesfully inhibit responses approximately 50% of the time by manipulating the stop-signal delays. Here, we are interested in whether there are any consistent brain responses to successfully inhibiting a motor response (i.e., **Successful Stop** trials) compared to trials in which no inhibition was required (i.e., **Go** trials). We will be ignoring the rest of the task for now.

    This exam is due by midnight on **Monday 5/11/2026**. Please upload it to canvas.

    You might consider using uvx to make sure that you are using nilearn version pinned to 0.13.1 so that downloading masks from the web and plotting work.

    `uvx marimo edit --sandbox Exam_Spring2026.py`

    I have included a few helper functions, which you might find useful.

    ## Accessing the data
    The stop-signal data we use here is publicly hosted on [OpenNeuro](https://openneuro.org/) — it's the dataset associated with [Aron, Behrens, Smith, Frank, & Poldrack (2007)](https://doi.org/10.1523/JNEUROSCI.0519-07.2007), accession **`ds000007`**. OpenNeuro stores every dataset on a public S3 bucket at `s3://openneuro.org/<dataset_id>` that anyone can read anonymously — no AWS account needed.

    The data are available on discovery `/dartfs/rc/lab/D/DBIC/DBIC/psych160/data/stopsignal`.

    If you prefer to work locally on your own computer, you can download from the open neuro or from this dropbox [link](/dartfs/rc/lab/D/DBIC/DBIC/psych160/data/stopsignal), which contains the preprocessed data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Univariate GLM
    Your task is to:
    1. Preprocess at least 1 subject using fmriprep. This can be done on discovery or locally on your own computer. I will provide preprocessed data for everyone in case you don't have enough time to preprocess more than 1 participant. **Include the fmriprep QC or screenshot of it with your exam for proof of completion**
    2. Analyze the data to produce a thresholded group analysis comparing Successful Stop to Go trials corre.
    3. Generate a plot of the thresholded group results
    4. Write up a short paragraph of what you did and what you found.
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nibabel as nib
    import pandas as pd
    from nltools.data import Brain_Data, Design_Matrix
    from nltools.mask import expand_mask, roi_to_brain
    from nltools.file_reader import onsets_to_dm
    from nltools.external import glover_hrf
    from nltools.stats import zscore
    from nilearn.plotting import view_img, glass_brain, plot_stat_map
    from tqdm import tqdm
    from dartbrains_tools import bids

    def make_motion_covariates(mc, tr=2.0):
        """Build motion covariates: z(mc), z(mc)^2, diff, diff^2."""
        z_mc = zscore(mc)
        all_mc = pd.concat([z_mc, z_mc ** 2, z_mc.diff(), z_mc.diff() ** 2], axis=1)
        all_mc.fillna(value=0, inplace=True)
        return Design_Matrix(all_mc, sampling_freq=1 / tr)



    # Wherever the BIDS root lives (Discovery, JupyterHub shared, or local)
    data_dir = "/Users/lukechang/Dropbox/Dartbrains/JupyterHub/data/stopsignal"
    task = 'stopsignal'

    tr = bids.get_tr(data_dir, task=task)

    sub_list = bids.get_subjects(data_dir)
    sub = sub_list[0]

    bold_path = bids.get_file(
      data_dir,
      subject=sub,
      scope="derivatives",
      suffix="bold",
      task=task,
      run=1,
    )

    cov = bids.load_confounds(data_dir, subject=sub, run=1, task=task)
    return Brain_Data, expand_mask, np, os, pd


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Connectivity

    Building on this task contrast, we might also want to map out if regions involved in cognitive control are communicating with other regions. To do this, you will perform a seed-based functional connectivity analysis to identify regions that co-activate with the inferior frontal gyrus/anterior insula (roi=37). This mask is available to download from [neurovault]('https://neurovault.org/media/images/8423/k50_2mm.nii.gz') or can be accessed on Discovery here `/dartfs/rc/lab/D/DBIC/DBIC/psych160/masks/k50_2mm.nii.gz`.

    Your goal is to:

    1. Identify how the ROI is functionally connected to the rest of the brain in this dataset.
    2. This will require you to extract signal from the seed mask
    3. Model brain activity at every voxel in the brain based on coactivations with this seed using appropriate covariates.
    4. Generate a plot of the thresholded group results
    5. Write up a short paragraph explaining what you did and include a figure.
    """)
    return


@app.cell
def _(Brain_Data, expand_mask, os):
    mask_dir = '/Users/lukechang/Dropbox/Dartbrains/JupyterHub/masks'

    mask = Brain_Data(os.path.join(mask_dir, 'k50_2mm.nii.gz'))
    mask_x = expand_mask(mask)
    mask_x[37].iplot()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cleaning up Noisy Data

    fMRI data is often very noisy and the primary goal of preprocessing is to remove the noise, but retain the true signal.

    In this problem, I have generated a timeseries signal that contains much of the noise that you will typically encounter with neuroimaging analysis. However, hidden within the noise is the true task-related signal that I have generated. Do your best to remove this noise using some of the techniques we have explored in the signal processing and GLM tutorials. However, be careful not to remove *too* much noise as I will be checking to see how well your denoised signal compares to the true signal.

    Load this file: `Noisy_Signal.csv`

    Your task is to:
    1. Denoise the data
    2. plot the noisy and denoised data so that we can see the difference of your denoising procedure
    3. write out your denoised data to a file so I can compare it to the original signal.
    4. Briefly describe the specific steps you used to remove the noise.
    """)
    return


@app.cell
def _(np, pd):
    from numpy.fft import fft, ifft, fftfreq
    from scipy.signal import butter, filtfilt
    from nltools.stats import regress
    from dartbrains_tools.notebook_utils import plot_timeseries


    def find_spikes(data, global_spike_cutoff=3, diff_spike_cutoff=3):
        """Identify spikes from a 1D timeseries.

        Args:
            data: 1D timeseries
            global_spike_cutoff: cutoff (in SDs) for spikes in the global signal,
                or None to skip.
            diff_spike_cutoff: cutoff (in SDs) for spikes in frame-to-frame
                differences, or None to skip.

        Returns:
            DataFrame with spike indicator regressors.
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

        outlier = pd.DataFrame([x + 1 for x in range(len(data))], columns=["TR"])
        if global_spike_cutoff is not None:
            for i, loc in enumerate(global_outliers):
                outlier["global_spike" + str(i + 1)] = 0
                outlier["global_spike" + str(i + 1)].iloc[int(loc)] = 1

        if diff_spike_cutoff is not None:
            for i, loc in enumerate(frame_outliers):
                outlier["diff_spike" + str(i + 1)] = 0
                outlier["diff_spike" + str(i + 1)].iloc[int(loc)] = 1
        return outlier.drop("TR", axis=1)

    _tr = 2.0

    _data = pd.read_csv('Noisy_Signal.csv')

    plot_timeseries(_data)
    return (find_spikes,)


@app.cell
def _(find_spikes):
    # 1. remove spikes
    spike_covariates = find_spikes(_data, global_spike_cutoff=3, diff_spike_cutoff=3)
    spike_covariates.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
