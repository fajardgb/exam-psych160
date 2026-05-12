# Imaging Methods (PSYC 160) Final Exam Writeup - Gabe Fajardo

I ran the analyses primarily on the HPC. To optimize the workflow and set up future analyses on larger datasets, I wrote different scripts for each analysis. For convenience, I've included PDFs of the marimo python scripts and their outputs in this [Google Drive folder](https://drive.google.com/drive/folders/195nbcFXfE1IZyFbkgx0s_JItXKYUDgCh?usp=drive_link), and I also include main output images in this writeup as well. The full scripts can also be found on the HPC at this path: `/dartfs/rc/lab/D/DBIC/DBIC/psych160/students/gabef/exam`

Note: these scripts were initially ran in my lab's shared space, but I copied them to the course directory in case there are any access restrictions. 

## 1. Preprocessing + univariate GLM

I first ran `fMRIPrep` on subs 01 and 02. The bash script can be found in the `preproc` folder. PDFs of the HTML QC output can be found in the Google Drive folder ([sub-01](https://drive.google.com/file/d/1nt3l0y8sOHE11Xo7dPPPnILPHp3kmVqB/view?usp=sharing) and [sub-02](https://drive.google.com/file/d/17IrLDPJJ0qfDc1xlo8LcAyO9pA7_oDCq/view?usp=sharing)).

For the 1st level GLM, I ran this individually for each subject using `Nilearn`. I first gathered key hyperparameters from the paper: TR=2s, smoothing FWHM=5, and a high-pass filter = 1/66. I tried to figure out which specific nuisance regressors they used in their design matrix, but this was a bit unclear. I chose to include the 6 basic motion confounds and the default cosine drift model implemented in the `FirstLevelModel` function. To get the events df in the right format, I first had to first remove the first row (duration=NaN) and remove spaces from the trial_type condition names. [[FULL SCRIPT](https://drive.google.com/file/d/1OOQjOIB41W-OKdIz02Rea0GF56517GsP/view?usp=sharing)]

Here is a plot of an example design matrix:

![design_matrix](/Users/gabe/Downloads/exam/univariateGLM/sub-01_design_matrix.png)

I then loaded the events, fMRI BOLD images, and confounds, and fit the GLM. Note, run-03 was missing for sub-12, so I excluded that run and ran the analysis for sub-12 on the remaining 2 runs.

For each subject, I computed the desired contrast of `successful_stop > go` (`output="effect_size"` to get betas), and saved the beta map. I then took these beta maps, 1 for each subject, and used the `SecondLevelModel` function, where I estimated the group-level effect via the `intercept` contrast, yielding a group-level Z-map.

The cluster-wise thresholding they used (GRFT) was not readily available in `Nilearn`, so I instead applied a voxel-wise threshold of `z>2.3` and set an arbitrary cluster size of `> 30 voxels`. I plotted both surface plots and stat_map plots, similar to the paper. All code can be seen in the exported PDF of the 2nd level marimo notebook. [[FULL SCRIPT](https://drive.google.com/file/d/1bz3lZi5VQ8joLBgP-rPOq8HCRAXIGK_E/view)]

Here are some plots:

![surf_plot](/Users/gabe/Downloads/exam/univariateGLM/output_level2/group_contrast-successful_stop_vs_go_zmap_thresholded.png)

![coronal_plot](/Users/gabe/Downloads/exam/univariateGLM/output_level2/group_contrast-successful_stop_vs_go_zmap_thresholded_y_coronal.png)

![medial_plot](/Users/gabe/Downloads/exam/univariateGLM/output_level2/group_contrast-successful_stop_vs_go_zmap_thresholded_saggital_medial.png)

I found similar results as reported in the paper (fig. 2c): significant activation for `succesful stops` (greater than `go`) in the right STN, right pre-SMA, IFC, right Globus Pallidus (GP), right parietal cortex, and right insula.

## 2. Connectivity

I first ran the connectivity analysis for each subject using the GLM-based approach discussed in class. I smoothed the fMRI data (FWHM=6), and then used the `load_confounds` function to load the appropriate nuisance regressors: full motion (24 params), wm + csf, and compcor components. I also included linear and quadriatic drift trends, as well as global and local spike regressors.

I then extracted and z-scored the timeseries for the anterior insula mask (seed37 from the k50 parcellation mask). This time series and all the nuisance regressors mentioned above were combined into a design matrix for each run. We got the seed37 contrast (controlling for the covariates) for each subject, giving us which voxels coactivate with the signal in the anterior insula. [[FULL SCRIPT](https://drive.google.com/file/d/1KrdwnUWRf0pHbxxwHyWCDshnPv8YJB0j/view?usp=sharing)]

![seed37_mask](/Users/gabe/Downloads/exam/connectivity/seed37_mask.png)

![seed37_timeseries](/Users/gabe/Downloads/exam/connectivity/seed37_timeseries.png)

Examples design matrices:

![design_matrices](/Users/gabe/Downloads/exam/connectivity/design_matrices.png)

I then ran the a 2nd level model on these beta maps to get a group-level zmap. This map had lots of activation at different thresholds, so I decided to also run a stricter TFCE permutation correction. [[FULL SCRIPT](https://drive.google.com/drive/u/0/folders/195nbcFXfE1IZyFbkgx0s_JItXKYUDgCh)]

Here is the resulting surface plot:

 ![2ndlevelplot](/Users/gabe/Downloads/exam/connectivity/output_level2/group_seed37_connectivity_tfce_surface.png)

As expected, we find significant coactivation in the bilateral anterior insula. We also see coactivation in the bilateral STS ant TPJ, as well as the mPFC, cingulate cortex, caudate nucleus, and thalamus.

## 3. Denoise

I used global and local spikes, discrete cosine transforms, and polynomial drift nuisance variables to regress out the noise in the signal. The denoised signal can be found in `Denoised_Signal.csv`. [[FULL SCRIPT](https://drive.google.com/file/d/1EhuwHbUQk0X6Wa0bBFodMWbD-u43vW5L/view?usp=sharing)]

![design_matrix](/Users/gabe/Downloads/exam/denoise/design_matrix.png)

From the plot below, we can clearly see that spikes were removed. We see the clear downwards trend removed as well. 

![denoised_signal](/Users/gabe/Downloads/exam/denoise/denoised_signal.png)

