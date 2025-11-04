import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from dotenv import load_dotenv
from scipy.ndimage import center_of_mass
from tqdm import tqdm


def generate_thumbnail(case_id, metadata, dataset_location):
    image_path = dataset_location / case_id / "dicoms"
    mask_path = dataset_location / case_id / "lesion_masks"
    image_files = sorted(list(image_path.rglob("*.dcm")))
    mask_files = sorted(list(mask_path.rglob("*.dcm")))
    vol = sitk.GetArrayFromImage(sitk.ReadImage(image_files))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_files)) if mask_files else None

    row = metadata[metadata.CaseID == case_id]
    midslice = mask.sum(axis=(1, 2)).argmax() if mask_files else vol.shape[0] // 2
    window, level = 300, 150
    vmin = level - window // 2
    vmax = level + window // 2

    fig, ax = plt.subplots(figsize=(2, 2))
    fig.set_facecolor("black")
    ax.imshow(vol[midslice], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{row['CaseID'].item()}: {row['Age'].item()} yrs")
    if mask_files:
        coords = center_of_mass(mask[midslice])
        coords = np.array(coords)[::-1]
        ann_str = f"{row['Subtype'].item()}: {row['LesionVolume(mL)'].item():2.1f} mL {row['LesionAttenuation(HU)'].item():2.0f} HU"
        ax.annotate(
            ann_str,
            coords,
            arrowprops=dict(facecolor="red", shrink=0.05),
            xytext=(10, 450),
            color="red",
        )
    ax.axis("off")
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, 1:]  # Remove alpha channel
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser(description="Create a montage of CT scan thumbnails.")
    parser.add_argument(
        "--num-cases",
        type=int,
        default=3000,
        help="The number of cases to include in the montage.",
    )
    parser.add_argument(
        "--num-columns",
        type=int,
        default=12,
        help="The number of columns in the montage grid.",
    )
    parser.add_argument(
        "--cases-per-montage",
        type=int,
        default=3000,
        help="The number of cases to include in each montage file.",
    )
    args = parser.parse_args()

    load_dotenv()
    dataset_location = Path(os.environ.get("DATA", None))
    metadata = pd.read_csv(dataset_location / "RST_3000.csv")

    plot_metadata = metadata.copy()
    for col in plot_metadata.columns:
        if plot_metadata[col].nunique() == 1:
            plot_metadata.pop(col)
    sns.pairplot(plot_metadata)
    plt.savefig("synthetic_ich_pairplot.png")

    summary_stats = metadata.describe().transpose()
    summary_stats = summary_stats[["mean", "std"]]
    summary_stats.to_csv("synthetic_ich_summary_statistics.csv")

    categorical_cols = ["Scanner", "ReconKernel", "Subtype"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(categorical_cols):
        sns.countplot(y=col, data=metadata, ax=axes[i])
    plt.tight_layout()
    plt.savefig("synthetic_ich_categorical_plots.png")

    categorical_summary = (
        metadata[categorical_cols].apply(lambda x: x.value_counts()).transpose()
    )
    categorical_summary.to_csv("synthetic_ich_categorical_summary.csv")

    all_case_ids = metadata.CaseID.unique()
    case_ids = [
        case_id
        for case_id in all_case_ids
        if len(list((dataset_location / case_id / "dicoms").rglob("*.dcm"))) > 0
    ][: args.num_cases]
    num_montages = int(np.ceil(len(case_ids) / args.cases_per_montage))

    for i in range(num_montages):
        start_idx = i * args.cases_per_montage
        end_idx = min((i + 1) * args.cases_per_montage, len(case_ids))
        montage_case_ids = case_ids[start_idx:end_idx]

        num_rows = int(np.ceil(len(montage_case_ids) / args.num_columns))
        fig, axes = plt.subplots(
            num_rows, args.num_columns, figsize=(args.num_columns * 2, num_rows * 2),
            gridspec_kw=dict(wspace=0, hspace=0)
        )
        axes = axes.flatten()
        for ax_idx, case_id in tqdm(enumerate(montage_case_ids), total=len(montage_case_ids)):
            ax = axes[ax_idx]
            img = generate_thumbnail(case_id, metadata, dataset_location)
            ax.imshow(img)
            ax.axis("off")

        for ax_idx in range(len(montage_case_ids), num_rows * args.num_columns):
            axes[ax_idx].axis("off")

        plt.tight_layout()
        if num_montages == 1:
            plt.savefig("synthetic_ich_preview.png", dpi=300)
        else:
            plt.savefig(f"synthetic_ich_preview_{i+1}.png", dpi=300)


if __name__ == "__main__":
    main()
