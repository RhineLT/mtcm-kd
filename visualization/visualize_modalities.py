import os

import nibabel as nib
import matplotlib.pyplot as plt

__all__ = ['visualize_modalities']

def visualize_modalities(flair, t1, t1ce, t2, slice_no):
    """Plot slice_no of each volume in a row."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=300)  # Increase DPI to 300

    for i, (img, title) in enumerate(zip([flair, t1ce, t2, t1], ["FLAIR", "T1ce", "T2", "T1"])):
        img = nib.load(img).get_fdata()
        axes[i].imshow(img[:, :, slice_no], cmap="gray")
        axes[i].set_title(title)

    plt.tight_layout()  # Improve the layout of the graph
    plt.show()
    
    ## save the figure
    fig.savefig(os.path.join(os.getcwd(), "modalities.png"), dpi=300)
    
    
if __name__ == "__main__":
    sample = "BraTS2021_00150"
    flair = os.path.join(os.getcwd(), "BraTS_Dataset", sample, f"{sample}_flair.nii.gz")
    t1 = os.path.join(os.getcwd(), "BraTS_Dataset",  sample, f"{sample}_t1.nii.gz")
    t1ce = os.path.join(os.getcwd(), "BraTS_Dataset",  sample, f"{sample}_t1ce.nii.gz")
    t2 = os.path.join(os.getcwd(), "BraTS_Dataset",  sample, f"{sample}_t2.nii.gz")
    slice_no = 70
    visualize_modalities(flair, t1, t1ce, t2, slice_no)