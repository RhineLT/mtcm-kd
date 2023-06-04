import os

import nibabel as nib
import matplotlib.pyplot as plt

__all__ = ['visualize_modalities']

def visualize_modalities(flair, t1, t1ce, t2, slice_no):
    """Plot slice_no of each volume in a row."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    for i, (img, title) in enumerate(zip([flair, t1ce, t2, t1], ["FLAIR", "T1ce", "T2", "T1"])):
        img = nib.load(img).get_fdata()
        axes[i].imshow(img[:, :, slice_no],cmap="gray")
        axes[i].set_title(title)

    plt.show()