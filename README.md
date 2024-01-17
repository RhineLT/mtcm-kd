# Multi-Teacher Cross-Modal Distillation with Cooperative Deep Supervision Fusion Learning for Unimodal Segmentation

Abstract
---
<p align="justify">
Accurate brain tumor segmentation is a labor-intensive and time-consuming task that requires automation to enhance its efficacy. Numerous techniques that incorporate convolutional neural networks and transformers have been proposed to automate this process, and they have demonstrated impressive results. However, these methods rely on a whole set of multi-modal magnetic resonance imaging (MRI) scans, which are often unavailable in realistic clinical settings, resulting in their sub-optimal performance. To improve the segmentation process on limited modalities, we have proposed a novel multi-teacher cross-modal knowledge distillation framework that uses the privileged multi-modal information during training while utilizing unimodal information for inference. We introduced two learning strategies for knowledge distillation: 1) performance-aware response-based knowledge distillation and 2) cooperative deep supervision fusion learning (CDSFL). In the first strategy, we dynamically calculate the confidence weight for each teacher model, so that the knowledge distillation is based on its performance. Whereas, the CDSFL module improves the learning capability of multi-teacher models by enabling them to learn from each other. Moreover, the fused information is distilled into the student model to improve its deep supervision. Our framework focuses on the unimodal segmentation of the $T_{1ce}$ MRI sequence, which is widely available in clinical settings, structurally similar to the $T_1$ modality, and contains sufficient information for segmentation. Extensive experiments on different datasets (such as BraTS-2018, 2019, 2020, and 2021) demonstrate that our framework outperforms previous methods on unimodal segmentation of the $T_{1ce}$ and $T_1$ modalities. For instance, our framework achieves dice scores of $91.56\%$ and $87.14\%$ on the BraTS-2021 dataset using $T_{1ce}$ modality for tumor core and whole tumor segmentation, respectively

  ---
  
Contributions
---
The contribution of our work is stated below:

  - Proposed a novel feature fusion module for cooperative deep supervision learning among teacher models that incorporates attention mechanisms to contextualize the feature maps and enable the teacher models to learn from each other. The fused knowledge is then distilled into the student model to improve its representation.
  - Introduced customized performance-aware response-based knowledge distillation scheme for the multi-teacher framework. This scheme dynamically allocates weights to each teacher model for distilling its knowledge based on its performance.
  - Proposed a novel multi-teacher cross-modal online knowledge distillation framework for unimodal brain tumor segmentation. The framework combines the concepts of performance-aware response-based knowledge distillation and cooperative deep supervision fusion learning to increase its learning capability. 
  - Performed extensive experiments and evaluated the framework on the BraTS datasets from 2018 to 2021. Our approach significantly outperforms previous state-of-the-art models in unimodal brain tumor segmentation. 

---

Dataset
---
This work uses four BraTS datasets from 2018-2021. The first three BraTS datasets can be downloaded from [here](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015), While the BraTS-2021 dataset is available [here](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1).

---
Setup
---
Install the requirements
```bash
pip install requriements.txt
```
and if in case there is still an issue installing the cudatoolkit and GPU version, then refer to the site [Pytorch](https://pytorch.org/get-started/locally/) or you can use the below commands for installing the Pytorch along with cudatoolkit
 
- Linux / Window
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  
- Mac
  ```bash
  pip3 install torch torchvision torchaudio
  ```

Update and customize the model's configuration by making the required changes in the `config.json` file and running the code using the below command
```bash
python run.py
```

For the inference of the model, make the necessary changes in the `inference.py` file, and then run the script: 
```bash
python inference.py
```
---

3D Visualization
---
For visualizing the 3D MRI volumes of the brain tumor, first, predict the 3D volumes of MRI by executing the below script: 
```bash
python visualize.py
```
- then refer to the repository for the 3D visualization of the volumes: [3D nii Visulalizer](https://github.com/adamkwolf/3d-nii-visualizer)
</p>

</p>


