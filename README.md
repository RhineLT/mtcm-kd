# Multi-Teacher Cross-Modal Distillation with Cooperative Deep Supervision Fusion Learning for Unimodal Segmentation

Abstract
---
<p align="justify">
Accurate brain tumor segmentation is a labor-intensive and time-consuming task that requires automation to enhance its efficacy. Recent advanced techniques have shown promising results; however, their dependency on extensive multimodal magnetic resonance imaging (MRI) data limits their practicality in clinical environments where such data may not be readily available. To address this, we propose a novel multi-teacher cross-modal knowledge distillation framework, which utilizes the privileged multimodal data during training while relying solely on unimodal data for inference. Our framework is tailored to the unimodal segmentation of the ${T_{1ce}}$ MRI sequence, which is prevalently available in clinical practice and structurally akin to the ${T_{1}}$ modality, providing ample information for the segmentation task. Our framework introduces two learning strategies for knowledge distillation (KD): 1) performance-aware response-based KD and 2) cooperative deep supervision fusion learning (CDSFL). The first strategy involves dynamically assigning confidence weights to each teacher model based on its performance, ensuring that the KD is performance-driven, and the CDSFL module augments the learning capabilities of the multi-teacher models by fostering mutual learning. Moreover, the fused information is distilled into the student model to improve its deep supervision. Extensive experiments on BraTS datasets demonstrate that our framework achieves promising unimodal segmentation results on the ${T_{1ce}}$ and ${T_{1}}$ modalities and outperforms previous state-of-the-art methods.
  
---
  
Contributions
---
The contribution of our work is stated below:

  - Proposed a novel CDSFL module for cooperative deep supervision learning among teacher models that incorporates attention mechanisms to contextualize the feature maps and enable the teacher models to learn from each other. The fused knowledge is then distilled to the student model to improve its representation.
  - Introduced customized performance-aware response-based KD scheme for the multi-teacher framework that dynamically allocates weights to each teacher model to distill its knowledge based on its performance.
  - Proposed a novel MTCM-KD framework for unimodal brain tumor segmentation. The framework combines the concepts of performance-aware response-based KD and CDSFL to increase its learning capability.
  - Performed extensive experiments and evaluated the framework on the BraTS datasets from 2018 to 2021. Our approach yields promising results and surpasses previous state-of-the-art models in unimodal brain tumor segmentation of $T_{1}$ and $T_{1ce}$ modalities.
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


