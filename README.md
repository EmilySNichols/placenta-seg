# Placenta-Seg
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23374110.svg)](https://doi.org/10.5281/zenodo.23374110)

**Pretrained nnU-Net v2 model for automated placenta segmentation in T2*-weighted MRI**

Placenta-Seg provides a pretrained **2D nnU-Net v2, five-fold ensemble** and a Python command-line application (`run.py`) for segmenting the placenta in T2*-weighted MRI. The application discovers appropriately named images in a BIDS-style dataset, prepares the input for nnU-Net, and writes binary placenta masks and metadata to a separate output directory.

The model was developed for **two-slice, multi-echo placental MRI** and trained on the **first echo**. It has not been externally validated on other acquisition protocols. **Research use only; not intended for clinical decision-making.**

> **Release status:** The Python inference workflow has been tested successfully on an Apple Silicon Mac using an independently installed copy of the exported model. Docker packaging and formal BIDS validation have not yet been completed.

## Model at a glance

| Property | Details |
| --- | --- |
| Framework | nnU-Net v2 |
| Configuration | 2D |
| Ensemble | Five trained folds (0–4) |
| Input | First echo of placenta T2*-weighted MRI; 3D NIfTI or 4D NIfTI with echoes along the fourth dimension |
| Output | Binary NIfTI segmentation: 0 = background; 1 = placenta |
| Training set | 190 scans from 103 participants |
| Independent test set | 35 scans from 18 participants |
| Training acquisition | Approximately 256 × 256 × 2 voxels; 1.37 × 1.37 × 8.00 mm voxel size |

## Installation

### 1. Create a Python environment

A dedicated virtual environment is recommended. Use a Python version compatible with your installed nnU-Net v2 and PyTorch releases.

```bash
git clone https://github.com/EmilySNichols/placenta-seg.git
cd placenta-seg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application requires `nnunetv2`, `nibabel`, and `numpy` (and nnU-Net's transitive dependencies, including PyTorch). Hardware acceleration depends on your PyTorch installation. CPU inference is supported; `mps` is available on compatible Apple Silicon systems, and `cuda` on appropriately configured NVIDIA systems.

Check the command-line interface:

```bash
python run.py --help
```

### 2. Download and install the model

Download the pretrained five-fold nnU-Net model from the Placenta-Seg v1.0.0 release.

The model is distributed as placenta_nnunet.zip and includes all five trained folds required for ensemble inference.

After downloading, set the nnU-Net results directory:

```bash
export nnUNet_results="$HOME/nnUNet_results"
mkdir -p "$nnUNet_results"
```

Install the pretrained model:

```bash
nnUNetv2_install_pretrained_model_from_zip /path/to/placenta_nnunet.zip

```

The model will be installed under Dataset001_placenta and can then be used directly with the Placenta-Seg inference application.

## Input data

The current application searches for NIfTI files with the acquisition entity `acq-placenta` and the filename suffix `T2star`, for example:

```text
my_bids_dataset/
├── dataset_description.json
└── sub-01/
    └── ses-01/
        └── anat/
            └── sub-01_ses-01_acq-placenta_T2star.nii.gz
```

The current naming convention is the one recognized by this application; **formal compliance with the BIDS specification has not yet been verified**.

- **3D input:** used as provided.
- **4D input:** the application extracts volume 0 along the fourth dimension. For multi-echo acquisitions, **echoes must be ordered along that dimension, with the first echo first**. Do not supply a 4D time series or a derived T2* map as though it were a multi-echo image.
- Images with acquisition characteristics substantially different from those used for training may perform differently.

## Run segmentation

Activate your environment and set `nnUNet_results` first.

**All matching participants:**

```bash
python run.py \
  /path/to/my_bids_dataset \
  /path/to/segmentation_output \
  participant \
  --device cpu
```

**One participant, with Apple Silicon acceleration:**

```bash
python run.py \
  /path/to/my_bids_dataset \
  /path/to/segmentation_output \
  participant \
  --participant-label 01 \
  --device mps
```

Participant labels should be supplied **without** the `sub-` prefix. Use `--device cuda` on a properly configured NVIDIA system. The application runs the five-fold model ensemble using nnU-Net v2's inference command.

To retain intermediate input and prediction files for troubleshooting, add `--keep-work-dir`.

## Outputs

For the example above, the application writes a segmentation and JSON sidecar resembling:

```text
segmentation_output/
├── dataset_description.json
└── sub-01/
    └── ses-01/
        └── anat/
            ├── sub-01_ses-01_acq-placenta_desc-placenta_mask.nii.gz
            └── sub-01_ses-01_acq-placenta_desc-placenta_mask.json
```

The NIfTI mask contains **0 (background)** and **1 (placenta)**. Outputs are intended to retain the spatial geometry of the input image. Inspect the resulting segmentation visually before using it in downstream analyses. Output naming and metadata are **BIDS-oriented**; derivative compliance still requires validation.

## Development and evaluation

The model was developed using 225 placental MRI scans. The training set comprised **190 scans from 103 participants**; the independent held-out test set comprised **35 scans from 18 participants**. Participants were assigned exclusively to either training or testing. Within the training set, five-fold cross-validation was also grouped by participant, preventing scans from the same participant from appearing in both training and validation within a fold.

The model used the **2D nnU-Net v2 configuration**, trained for **1,000 epochs per fold**. First-echo images were paired with manually delineated binary placenta masks. During dataset preparation, mask spatial metadata were corrected to match the corresponding images without resampling their voxel arrays.

### Independent test-set performance

The five-fold ensemble was evaluated against manual reference masks on **35 held-out scans**. Values below are **mean ± sample standard deviation across scans**, rather than pooled-voxel metrics.

| Metric | Mean ± SD |
| --- | ---: |
| Dice similarity coefficient | **0.950 ± 0.034** |
| Jaccard index | 0.907 ± 0.059 |
| Sensitivity | 0.953 ± 0.053 |
| Specificity | 0.988 ± 0.018 |
| Precision | 0.952 ± 0.055 |
| 95th-percentile Hausdorff distance (HD95) | 2.48 ± 5.38 mm |
| Average symmetric surface distance (ASSD) | 0.37 ± 0.61 mm |

Median Dice was **0.963** (range **0.864–0.986**). The mean **signed** predicted-minus-reference volume difference was **+1.35 ± 35.90 mL** (mean percentage difference **+0.60 ± 9.78%**); this is not an absolute volume-error measure.

Surface-distance measures should be interpreted in light of the anisotropic acquisition (~8 mm through-plane slice thickness). These are **internal held-out test results**, not external validation.

### Inference smoke test

The exported model ZIP was installed into a separate nnU-Net results directory on another computer, and `run.py` successfully produced a mask. The checked example had input and mask dimensions of **256 × 256 × 2**, labels **[0, 1]**, matching image/mask affines and spatial dimensions, and a visually plausible segmentation. This confirms basic end-to-end operation in that environment, **not** compatibility across all datasets or systems.

## Limitations

- Trained for a specific, limited-coverage, two-slice placenta MRI acquisition; external generalization has not been established.
- Expects first-echo contrast; later echoes, quantitative T2* maps, and other modalities are not equivalent inputs.
- The current application has not yet been Docker-packaged or formally validated against the BIDS specification.
- Automated segmentations require visual quality control, especially for images with different contrast, resolution, anatomy, or artifacts.
- **Not a medical device and not intended for clinical use.**

## License

**Non-commercial use only.** Different licenses apply to the original application code and pretrained model:

- **Original application code:** [PolyForm Noncommercial License 1.0.0](LICENSE-CODE.md).
- **Pretrained model weights and repository documentation:** [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE-MODEL.md).
- **Third-party dependencies:** Retain their own licenses; these terms do not relicense nnU-Net, PyTorch, or other dependencies.

Commercial use requires separate permission from the relevant rights holder. Academic affiliation does not automatically establish that a particular use is non-commercial. See the linked license texts for the controlling terms.

## Citation

## Citation

If you use Placenta-Seg or its pretrained nnU-Net model in your research, please cite:

Nichols, E. (2026). Placenta-Seg: Automated Placenta Segmentation from T2-Weighted MRI* (v1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23374110

You can also generate a formatted citation using GitHub's Cite this repository option.

For reproducibility, please cite the specific software version used in your analysis.

The pretrained model and inference application are distributed for non-commercial use under their respective licenses. See `LICENSE-MODEL.md` and `LICENSE-CODE.md` for details.

Please also cite nnU-Net:

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods. 2021;18:203–211. https://doi.org/10.1038/s41592-020-01008-z

## Contact

For questions, bug reports, and non-commercial-use clarifications, please open a GitHub issue. For commercial licensing inquiries, contact Emily Nichols at enicho4@uwo.ca.
