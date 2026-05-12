# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.10.9",
#     "nilearn>=0.13.1",
#     "numpy>=2.4.4",
#     "pandas>=3.0.2",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import os

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from nilearn.image import load_img, math_img
    from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
    from nilearn.glm import threshold_stats_img
    from nilearn.reporting import get_clusters_table

    from nilearn.plotting import plot_img_on_surf, plot_stat_map, view_img

    import nilearn
    print(f"nilearn version: {nilearn.__version__}")
    return (
        SecondLevelModel,
        load_img,
        mo,
        np,
        os,
        pd,
        plot_img_on_surf,
        plt,
        threshold_stats_img,
        view_img,
    )


@app.cell
def _():
    N_SUBS = 13
    CONTRAST = "seed37_connectivity"
    return CONTRAST, N_SUBS


@app.cell
def _(N_SUBS, os):
    current_dir = os.getcwd()
    save_dir = f"{current_dir}/output_level2"
    os.makedirs(save_dir, exist_ok=True)

    beta_map_dir = f"{current_dir}/output"

    subs = [str(i).zfill(2) for i in range(1, 16)]
    subs.remove("08")
    subs.remove("11")
    assert len(subs) == N_SUBS
    return beta_map_dir, save_dir, subs


@app.cell
def _(CONTRAST, beta_map_dir, load_img, subs):
    beta_maps = []
    for sub in subs:
        beta_maps.append(load_img(f"{beta_map_dir}/sub-{sub}/sub-{sub}_{CONTRAST}_beta.nii.gz"))

    print(f"Loaded beta maps for {len(beta_maps)} subjects.")
    return (beta_maps,)


@app.cell
def _(beta_maps, np, subs):
    # make sure all beta maps same shape and affine
    shape1, affine1 = beta_maps[0].shape, beta_maps[0].affine
    for i, bm in enumerate(beta_maps):
        if bm.shape != shape1 or not np.allclose(bm.affine, affine1):
            print(f"Warning: beta map for sub-{subs[i]} has different shape or affine.")
            print(f"  shape: {bm.shape} vs {shape1}")
            print(f"  affine:\n{bm.affine}\nvs\n{affine1}")
    return


@app.cell
def _(CONTRAST, SecondLevelModel, beta_maps, pd, save_dir):
    design_matrix = pd.DataFrame([1] * len(beta_maps), columns=["intercept"])

    second_level_model = SecondLevelModel(smoothing_fwhm=None, n_jobs=-1)
    second_level_model.fit(
        beta_maps,
        design_matrix=design_matrix,
    )
    zmap = second_level_model.compute_contrast(
        second_level_contrast="intercept",
        output_type="z_score",
    )
    zmap.to_filename(f"{save_dir}/group_{CONTRAST}_zmap.nii.gz")
    return (zmap,)


@app.cell
def _(view_img, zmap):
    view_img(zmap, symmetric_cmap=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Threshold
    """)
    return


@app.cell
def _(threshold_stats_img, view_img, zmap):
    thresholded_map, threshold = threshold_stats_img(
        zmap,
        alpha=0.01,                 # p < 0.05
        height_control=None,        # voxel-level control - if None, use threshold parameter
        cluster_threshold=30,       # min cluster size - arbitrary... since GRFT cluster-threshold used in paper not available
        two_sided=True,            # one-sided test (z>2.3)
        threshold=3.09               # voxel-wise z threshold
    )
    view_img(thresholded_map)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TFCE
    """)
    return


@app.cell
def _():
    # # run TFCE
    # out_dict = non_parametric_inference(
    #     second_level_input=beta_maps,          # list of Nifti1Image or paths
    #     design_matrix=design_matrix,
    #     model_intercept=True,                  # one-sample test
    #     n_perm=5000,
    #     two_sided_test=True,                   # set False for directional hypothesis
    #     tfce=True,                             # ← enables TFCE
    #     threshold=None,                        # not needed when tfce=True
    #     smoothing_fwhm=None,                   # smooth here if not done at 1st level
    #     n_jobs=6,                             # parallelise across all available cores
    #     verbose=1,
    #     random_state=42,
    # )
    return


@app.cell
def _():
    # t_map        = out_dict["t"]
    # tfce_scores  = out_dict["tfce"]
    # logp_t       = out_dict["logp_max_t"]        # voxelwise FWE
    # logp_tfce    = out_dict["logp_max_tfce"]     # TFCE-FWE  ← threshold at log10(20) ≈ 1.3 for p<0.05

    # # save all maps
    # t_map.to_filename(f"{save_dir}/group_{CONTRAST}_tmap.nii.gz")
    # tfce_scores.to_filename(f"{save_dir}/group_{CONTRAST}_tfce.nii.gz")
    # logp_t.to_filename(f"{save_dir}/group_{CONTRAST}_logp_t.nii.gz")
    # logp_tfce.to_filename(f"{save_dir}/group_{CONTRAST}_logp_tfce.nii.gz")
    return


@app.cell
def _(CONTRAST, load_img, np, save_dir, view_img):
    # load and view TFCE map
    t_map = load_img(f"{save_dir}/group_{CONTRAST}_tmap.nii.gz")
    tfce_scores = load_img(f"{save_dir}/group_{CONTRAST}_tfce.nii.gz")
    logp_tfce = load_img(f"{save_dir}/group_{CONTRAST}_logp_tfce.nii.gz")

    THRESHOLD =-np.log10(0.05) # 1.3
    view_img(logp_tfce, symmetric_cmap=False, threshold=THRESHOLD)
    return THRESHOLD, logp_tfce


@app.cell
def _(CONTRAST, THRESHOLD, logp_tfce, plot_img_on_surf, plt, save_dir):
    # surface plot
    plot_img_on_surf(
        logp_tfce,
        surf_mesh="fsaverage5",
        views=["lateral", "medial"],
        hemispheres=["left", "right"],
        colorbar=True,
        cmap='inferno',
        symmetric_cmap=False,
        threshold=float(THRESHOLD),
        inflate=True,
        bg_on_data=False,
        title=f"{CONTRAST} (TFCE corrected, p<0.05)"
    )
    plt.savefig(f"{save_dir}/group_{CONTRAST}_tfce_surface.png", dpi=300)
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
    I first ran the connectivity analysis for each subject using the GLM-basde approach discussed in class. I smoothed the fMRI data (FWHM=6), and then used the `load_confounds` function to load the appropriate nuisance regressors: full motion (24 params), wm + csf, and compcor components. I also included linear and quadriatic drift trends, as well as global and local spike regressors.

    I then extracted and z-scored the timeseries for the anterior insula mask (seed37 from the k50 parcellation mask). This time series and all the nuisance regressors mentioned above were combined into a design matrix for each run. We got the seed37 contrast (controlling for the covariates) for each subject, giving us which voxels coactivate with the signal in the anterior insula.

    I then ran the a 2nd level model on these beta maps to get a group-level zmap. This map had lots of activation at different thresholds, so I decided to also run a stricter TFCE permutation correction. Here is the resulting surface plot:

     ![2ndlevelplot](output_level2/group_seed37_connectivity_tfce_surface.png)


    As expected, we find significant coactivation in the bilateral anterior insula. We also see coactivation in the bilateral STS ant TPJ, as well as the mPFC, cingulate cortex, caudate nucleus, and thalamus.
    """)
    return


if __name__ == "__main__":
    app.run()
