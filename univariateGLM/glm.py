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


__generated_with = "0.23.5"

# %%
import pandas as pd
import os
import time
import sys

import marimo as mo

from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.image import load_img

import nilearn
print(f"nilearn version: {nilearn.__version__}")

# %%
mo.md(r"""
## Set up directories and hyperparameters
""")

# %%
# Hyperparameters
TR = 2.0
FWHM = 5.0
HIGH_PASS = 1/66

num_runs = 3
runs = [str(i + 1).zfill(2) for i in range(num_runs)]

# sub ID from command line
sub = sys.argv[1]
sub = sub.zfill(2)  # zero-pad to 2 digits if needed
# sub = "01"

if sub != "01":
    time.sleep(10)  # Sleep for 10 seconds if not sub-001 - for creating save dirs

print(f"Running sub-{sub}")
print(f"Hyperparameters: TR={TR}, FWHM={FWHM}, HIGH_PASS={HIGH_PASS:.6f}, runs={runs}")

# %%
# Set directories

# data_dir = f"../data/fmriprep/sub-{sub}/func"
data_dir = f"/dartfs/rc/lab/D/DBIC/DBIC/psych160/data/stopsignal/derivatives/fmriprep/sub-{sub}/func"
current_dir = os.getcwd()
save_dir = f"{current_dir}/output/sub-{sub}"
os.makedirs(save_dir, exist_ok=True)

report_save_dir = f"{save_dir}/reports"
os.makedirs(report_save_dir, exist_ok=True)

# %%
mo.md(r"""
## Load the data for this sub
""")

# %%
fmri_imgs = []
events = []
confounds = []

for run in runs:

    if sub == "12" and run == "03":
        print(f"Skipping sub-{sub} run-{run} due to missing run.")
        continue
    events_df = pd.read_csv(f"{data_dir}/sub-{sub}_task-stopsignal_run-{run}_events.tsv", sep="\t")
    events_df = events_df.dropna(subset=['duration'])
    events_df['trial_type'] = events_df['trial_type'].str.replace(' ', '_')
    events_df = events_df[["onset", "duration", "trial_type"]]
    events.append(events_df)

    fmri_img_path = f"{data_dir}/sub-{sub}_task-stopsignal_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    fmri_img = load_img(fmri_img_path)
    fmri_imgs.append(fmri_img)

    confound, _ = load_confounds(fmri_img_path, 
                                 strategy=["motion"],
                                 motion="basic")
    confounds.append(confound)

assert len(events) == len(fmri_imgs) == len(confounds), f"Mismatch in events ({len(events)}) or fMRI images ({len(fmri_imgs)}) or confounds ({len(confounds)})."
print(f"Loaded {len(events)} runs of data for sub-{sub}.")

# %%
mo.md(r"""
## Fit the GLM
""")

# %%
glm = FirstLevelModel(t_r=TR, high_pass=HIGH_PASS, smoothing_fwhm=FWHM, n_jobs=-1)
glm.fit(fmri_imgs, events=events, confounds=confounds)

# %%
# compute desired contrasts
contrasts = ["successful_stop - go"]

for c in contrasts:
    beta_map = glm.compute_contrast(c, output_type="effect_size")
    beta_map.to_filename(f"{save_dir}/sub-{sub}_contrast-{c.replace(' ', '_').replace('-', 'vs')}_beta.nii.gz")
print(f"Computed contrasts and saved beta maps for sub-{sub} to {save_dir}.")

# %%
# save GLM report
report = glm.generate_report(title=f"GLM Report for sub-{sub}", contrasts=contrasts)
report.save_as_html(f"{report_save_dir}/sub-{sub}_glm_report.html")
print(f"Saved GLM report for sub-{sub} to {report_save_dir}.")