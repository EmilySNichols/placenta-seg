#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# APP SETTINGS
# ============================================================

APP_NAME = "placenta-seg"
APP_VERSION = "1.0.0"

# This must match the trained nnU-Net dataset/model
NNUNET_DATASET = "Dataset001_placenta"
NNUNET_CONFIGURATION = "2d"

# Expected BIDS naming:
# sub-01_ses-01_acq-placenta_T2star.nii.gz
DEFAULT_SUFFIX = "T2star"
DEFAULT_ACQUISITION = "placenta"


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def get_parser():

    parser = argparse.ArgumentParser(
        description=(
            "BIDS App for automatic placenta segmentation "
            "from placenta T2* MRI using a trained nnU-Net v2 model."
        )
    )

    parser.add_argument(
        "bids_dir",
        type=Path,
        help="Root directory of the input BIDS dataset.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for BIDS derivatives.",
    )

    parser.add_argument(
        "analysis_level",
        choices=["participant"],
        help="Analysis level. Currently only 'participant' is supported.",
    )

    parser.add_argument(
        "--participant-label",
        nargs="+",
        default=None,
        help=(
            "Participant label(s) to process, without the 'sub-' prefix. "
            "Example: --participant-label 01 02 03. "
            "If omitted, all participants are processed."
        ),
    )

    parser.add_argument(
        "--session-label",
        nargs="+",
        default=None,
        help=(
            "Optional session label(s) to process, without the 'ses-' prefix. "
            "Example: --session-label 01 02."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help=(
            "Device used for nnU-Net prediction. "
            "Options: cpu, cuda, mps. Default: cpu."
        ),
    )

    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help=(
            f"BIDS suffix to process. Default: {DEFAULT_SUFFIX}. "
            "Expected filenames end in '_T2star.nii.gz'."
        ),
    )

    parser.add_argument(
        "--acquisition",
        default=DEFAULT_ACQUISITION,
        help=(
            f"BIDS acquisition label to process. "
            f"Default: {DEFAULT_ACQUISITION}. "
            "Expected filenames contain '_acq-placenta_'."
        ),
    )

    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help=(
            "Keep temporary nnU-Net input and output files "
            "inside the derivatives output directory."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )

    return parser


# ============================================================
# BIDS FILENAME HELPERS
# ============================================================

def strip_nii_extension(filename):
    """
    Remove .nii or .nii.gz extension.
    """

    filename = str(filename)

    if filename.endswith(".nii.gz"):
        return filename[:-7]

    if filename.endswith(".nii"):
        return filename[:-4]

    return filename


def parse_bids_entities(filename):
    """
    Parse BIDS entities from a filename.

    Example:

        sub-01_ses-01_acq-placenta_T2star.nii.gz

    returns:

        {
            "sub": "01",
            "ses": "01",
            "acq": "placenta",
            "suffix": "T2star"
        }
    """

    stem = strip_nii_extension(Path(filename).name)

    parts = stem.split("_")

    if len(parts) < 2:
        return {}

    entities = {}

    # Last component is the suffix
    entities["suffix"] = parts[-1]

    # Everything before the suffix may contain BIDS entities
    for part in parts[:-1]:

        if "-" not in part:
            continue

        key, value = part.split("-", 1)

        if key and value:
            entities[key] = value

    return entities


# ============================================================
# BIDS IMAGE DISCOVERY
# ============================================================

def find_bids_images(
    bids_dir,
    participants=None,
    sessions=None,
    suffix=DEFAULT_SUFFIX,
    acquisition=DEFAULT_ACQUISITION,
):
    """
    Find placenta T2* MRI scans in the BIDS dataset.

    Expected filename format:

        sub-01_ses-01_acq-placenta_T2star.nii.gz

    Selection rules:
        - must have suffix T2star
        - must have acq-placenta
        - derivatives are ignored
        - participant/session filters are optional
    """

    candidates = []

    # Support both .nii.gz and .nii
    nifti_files = list(bids_dir.rglob("*.nii.gz"))
    nifti_files += list(bids_dir.rglob("*.nii"))

    for nii in nifti_files:

        try:
            relative = nii.relative_to(bids_dir)
        except ValueError:
            continue

        # Never process derivative outputs as raw input
        if "derivatives" in relative.parts:
            continue

        entities = parse_bids_entities(nii.name)

        if not entities:
            continue

        # Must match desired suffix
        if entities.get("suffix") != suffix:
            continue

        # Must match placenta acquisition
        if entities.get("acq") != acquisition:
            continue

        participant = entities.get("sub")
        session = entities.get("ses")

        # Subject is required
        if participant is None:
            continue

        # Optional participant filtering
        if participants is not None:
            if participant not in participants:
                continue

        # Optional session filtering
        if sessions is not None:
            if session not in sessions:
                continue

        candidates.append(
            {
                "path": nii,
                "entities": entities,
            }
        )

    return sorted(
        candidates,
        key=lambda x: str(x["path"])
    )


# ============================================================
# NNUNET CASE NAMING
# ============================================================

def make_case_id(entities):
    """
    Create a unique nnU-Net case identifier.

    Example:

        sub-01_ses-01_acq-placenta_T2star.nii.gz

    becomes:

        sub-01_ses-01_acq-placenta
    """

    pieces = [
        f"sub-{entities['sub']}"
    ]

    if "ses" in entities:
        pieces.append(
            f"ses-{entities['ses']}"
        )

    if "acq" in entities:
        pieces.append(
            f"acq-{entities['acq']}"
        )

    if "run" in entities:
        pieces.append(
            f"run-{entities['run']}"
        )

    return "_".join(pieces)


# ============================================================
# PREPARE INPUT FOR NNUNET
# ============================================================

def prepare_nnunet_input(source_file, output_file):
    """
    Convert a BIDS T2* image into the single-channel NIfTI
    expected by the trained nnU-Net model.

    If the input is 3D:
        - use it directly

    If the input is 4D:
        - select volume 0, corresponding to the first echo

    The affine, qform, and sform are preserved.
    """

    img = nib.load(source_file)

    data = np.asarray(img.dataobj)

    print(f"  Input shape: {data.shape}")

    if data.ndim == 3:

        print("  3D image detected.")
        print("  Using image directly.")

        output_data = data

    elif data.ndim == 4:

        if data.shape[3] < 1:
            raise ValueError(
                f"4D image contains no volumes: {source_file}"
            )

        print("  4D image detected.")
        print(
            f"  Number of volumes/echoes: {data.shape[3]}"
        )
        print(
            "  Selecting first volume (index 0), "
            "matching model training."
        )

        output_data = data[:, :, :, 0]

    else:

        raise ValueError(
            f"Unsupported image dimensions for {source_file}: "
            f"{data.shape}"
        )

    # Create output image using source affine
    output_img = nib.Nifti1Image(
        output_data,
        img.affine
    )

    # Explicitly preserve qform
    qform, qcode = img.get_qform(coded=True)

    if qform is not None:
        output_img.set_qform(
            qform,
            int(qcode)
        )

    # Explicitly preserve sform
    sform, scode = img.get_sform(coded=True)

    if sform is not None:
        output_img.set_sform(
            sform,
            int(scode)
        )

    output_img.set_data_dtype(
        output_data.dtype
    )

    nib.save(
        output_img,
        output_file
    )

    print(f"  nnU-Net input: {output_file}")


# ============================================================
# RUN NNUNET
# ============================================================

def run_nnunet(
    input_dir,
    output_dir,
    device,
):
    """
    Run nnUNetv2_predict.

    Because all five trained folds are present, nnU-Net will
    ensemble them during prediction.
    """

    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        NNUNET_DATASET,
        "-c",
        NNUNET_CONFIGURATION,
        "-device",
        device,
    ]

    print("\n================================")
    print("RUNNING NNUNET")
    print("================================\n")

    print("Command:")
    print(" ".join(command))
    print()

    subprocess.run(
        command,
        check=True,
    )


# ============================================================
# BIDS DERIVATIVE NAMING
# ============================================================

def make_derivative_filename(entities):
    """
    Create output filename.

    Example:

        sub-01_ses-01_acq-placenta_T2star.nii.gz

    becomes:

        sub-01_ses-01_acq-placenta_desc-placenta_mask.nii.gz
    """

    pieces = [
        f"sub-{entities['sub']}"
    ]

    if "ses" in entities:
        pieces.append(
            f"ses-{entities['ses']}"
        )

    if "acq" in entities:
        pieces.append(
            f"acq-{entities['acq']}"
        )

    if "run" in entities:
        pieces.append(
            f"run-{entities['run']}"
        )

    pieces.append(
        "desc-placenta"
    )

    pieces.append(
        "mask"
    )

    return "_".join(pieces) + ".nii.gz"


# ============================================================
# JSON SIDECAR
# ============================================================

def make_bids_source_reference(
    source_file,
    bids_dir,
):
    """
    Create a relative BIDS source reference.
    """

    try:
        relative = source_file.relative_to(
            bids_dir
        )

        return "bids::" + str(relative)

    except ValueError:

        return str(source_file)


def write_sidecar(
    output_file,
    source_file,
    bids_dir,
):
    """
    Write metadata describing the segmentation.
    """

    json_file = Path(
        strip_nii_extension(output_file)
        + ".json"
    )

    metadata = {
        "Description": (
            "Automatic binary placenta segmentation "
            "generated from placenta T2* MRI."
        ),
        "Type": "Mask",
        "Label": "placenta",
        "Sources": [
            make_bids_source_reference(
                source_file,
                bids_dir,
            )
        ],
        "GeneratedBy": [
            {
                "Name": APP_NAME,
                "Version": APP_VERSION,
                "Description": (
                    "Placenta segmentation using a "
                    "trained nnU-Net v2 2D model."
                ),
            }
        ],
    }

    with open(json_file, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )


# ============================================================
# COPY PREDICTION TO DERIVATIVE DATASET
# ============================================================

def copy_prediction_to_derivatives(
    prediction,
    entities,
    source_file,
    bids_dir,
    output_root,
):
    """
    Copy nnU-Net output into a BIDS derivative hierarchy.
    """

    participant = entities["sub"]
    session = entities.get("ses")

    output_dir = (
        output_root
        / f"sub-{participant}"
    )

    if session is not None:

        output_dir = (
            output_dir
            / f"ses-{session}"
        )

    # Keep placenta T2* derivative under anat
    output_dir = (
        output_dir
        / "anat"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = make_derivative_filename(
        entities
    )

    destination = (
        output_dir
        / filename
    )

    shutil.copy2(
        prediction,
        destination,
    )

    write_sidecar(
        output_file=destination,
        source_file=source_file,
        bids_dir=bids_dir,
    )

    return destination


# ============================================================
# DATASET_DESCRIPTION.JSON
# ============================================================

def write_dataset_description(
    output_dir,
):
    """
    Write dataset_description.json for the derivative dataset.
    """

    description = {
        "Name": "Placenta segmentation derivatives",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": APP_NAME,
                "Version": APP_VERSION,
                "Description": (
                    "Automatic placenta segmentation using "
                    "a trained nnU-Net v2 model."
                ),
            }
        ],
    }

    path = (
        output_dir
        / "dataset_description.json"
    )

    with open(path, "w") as f:

        json.dump(
            description,
            f,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = get_parser().parse_args()

    bids_dir = args.bids_dir.resolve()

    output_dir = args.output_dir.resolve()

    # --------------------------------------------------------
    # Basic BIDS dataset check
    # --------------------------------------------------------

    if not bids_dir.exists():

        raise FileNotFoundError(
            f"BIDS directory does not exist:\n"
            f"{bids_dir}"
        )

    dataset_description = (
        bids_dir
        / "dataset_description.json"
    )

    if not dataset_description.exists():

        raise RuntimeError(
            f"{bids_dir} does not appear to be "
            "a BIDS dataset because "
            "dataset_description.json is missing."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("================================")
    print("PLACENTA SEGMENTATION BIDS APP")
    print("================================")
    print()

    print(f"Version: {APP_VERSION}")

    print(
        f"\nInput BIDS dataset:\n"
        f"  {bids_dir}"
    )

    print(
        f"\nOutput derivatives directory:\n"
        f"  {output_dir}"
    )

    print(
        f"\nExpected acquisition:\n"
        f"  acq-{args.acquisition}"
    )

    print(
        f"\nExpected suffix:\n"
        f"  {args.suffix}"
    )

    # --------------------------------------------------------
    # Find input images
    # --------------------------------------------------------

    images = find_bids_images(
        bids_dir=bids_dir,
        participants=args.participant_label,
        sessions=args.session_label,
        suffix=args.suffix,
        acquisition=args.acquisition,
    )

    print()
    print(
        f"Found {len(images)} eligible "
        "placenta T2* image(s)."
    )

    if len(images) == 0:

        raise RuntimeError(
            "\nNo eligible placenta T2* images found.\n\n"
            "Expected filenames similar to:\n"
            "  sub-01_ses-01_acq-placenta_T2star.nii.gz\n"
        )

    print()

    for item in images:
        print(
            f"  {item['path']}"
        )

    # --------------------------------------------------------
    # Set up temporary working directory
    # --------------------------------------------------------

    temporary = None

    if args.keep_work_dir:

        work_dir = (
            output_dir
            / "work"
        )

        if work_dir.exists():

            shutil.rmtree(
                work_dir
            )

        work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    else:

        temporary = tempfile.TemporaryDirectory(
            prefix="placenta_seg_"
        )

        work_dir = Path(
            temporary.name
        )

    nnunet_input = (
        work_dir
        / "nnunet_input"
    )

    nnunet_output = (
        work_dir
        / "nnunet_output"
    )

    nnunet_input.mkdir(
        parents=True,
        exist_ok=True,
    )

    nnunet_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Prepare images for nnU-Net
    # --------------------------------------------------------

    case_lookup = {}

    print()
    print("================================")
    print("PREPARING NNUNET INPUT")
    print("================================")

    for item in images:

        source = item["path"]

        entities = item["entities"]

        case_id = make_case_id(
            entities
        )

        if case_id in case_lookup:

            raise RuntimeError(
                f"Duplicate nnU-Net case ID generated:\n"
                f"  {case_id}\n\n"
                "Check whether multiple input images have "
                "identical subject/session/acquisition/run entities."
            )

        nnunet_file = (
            nnunet_input
            / f"{case_id}_0000.nii.gz"
        )

        print()
        print(
            f"Preparing {case_id}"
        )

        print(
            f"  Source: {source}"
        )

        prepare_nnunet_input(
            source_file=source,
            output_file=nnunet_file,
        )

        case_lookup[case_id] = item

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    run_nnunet(
        input_dir=nnunet_input,
        output_dir=nnunet_output,
        device=args.device,
    )

    # --------------------------------------------------------
    # Copy predictions into BIDS derivatives
    # --------------------------------------------------------

    print()
    print("================================")
    print("WRITING BIDS DERIVATIVES")
    print("================================")
    print()

    n_written = 0

    for case_id, item in case_lookup.items():

        prediction = (
            nnunet_output
            / f"{case_id}.nii.gz"
        )

        if not prediction.exists():

            print(
                f"WARNING: prediction missing "
                f"for {case_id}"
            )

            continue

        destination = (
            copy_prediction_to_derivatives(
                prediction=prediction,
                entities=item["entities"],
                source_file=item["path"],
                bids_dir=bids_dir,
                output_root=output_dir,
            )
        )

        print(
            f"Written:\n  {destination}"
        )

        n_written += 1

    # --------------------------------------------------------
    # Derivative dataset metadata
    # --------------------------------------------------------

    write_dataset_description(
        output_dir
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("================================")
    print("PLACENTA SEGMENTATION COMPLETE")
    print("================================")

    print(
        f"\nImages processed: {len(images)}"
    )

    print(
        f"Segmentations written: {n_written}"
    )

    print(
        f"\nResults:\n  {output_dir}"
    )

    if args.keep_work_dir:

        print(
            f"\nWorking files retained:\n"
            f"  {work_dir}"
        )

    else:

        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
