![](title.png)
#  FunPiQ: A New Benchmark for Pixel-Level Quality Assessment in Fundus Images
> Abstract: Color fundus photography (CFP) is the most common ophthalmic imaging modality for large-scale screening. However, it is highly susceptible to degradations, making robust fundus image quality assessment (FIQA) crucial. The criteria for what constitutes high-quality at the image level vary across clinical tasks, making FIQA dependent on expert knowledge. This motivated the development of automated methods and datasets. While existing datasets aim to standardize image-level quality, their criteria often differ. Furthermore, image-level labels preclude the quantitative evaluation of localized degradations, which is essential for trustworthy FIQA. We argue that pixel-level FIQA based on anatomical visibility represents a more task-agnostic, explainable approach. In this work, we introduce FunPiQ, the first FIQA benchmark to provide pixel-level quality annotations. In addition, we propose EFIQA-CP, an explainable-by-design (EBD) method that uses quality pseudo-labels based on anatomical visibility to train a CNN via Non-Negative Positive-Unlabeled learning. Extensive evaluations of classification methods with post-hoc explanations, anomaly detection methods, and EBD methods demonstrate the superior performance of the latter and, particularly, of EFIQA-CP.

## Download the dataset
The annotation could be downloaded in [release](https://github.com/MICCAI2026-3819/FunPiQ/releases/tag/v0.0.0).

The dataset could be downloaded in here:
[EyeQ](https://github.com/HzFu/EyeQ?tab=readme-ov-file), 
[BRSET](https://physionet.org/content/brazilian-ophthalmological/1.0.1/), 
[mBRSET](https://physionet.org/content/mbrset/1.0/). For EyeQ, please follow the official preprocessing script to get the resized image. For BRSET and mBRSET, no preprocessing is needed.

## Annotation Format
The annotation is stored as Indexed PNG. You can read it with PIL:
```python
from PIL import Image
import numpy as np

# Load the indexed PNG mask
mask_path = 'path/to/annotation.png'
pil_mask = Image.open(mask_path)

# Convert to a numpy array where values are the exact class indices
mask_array = np.array(pil_mask)

# (Optional) Retrieve the exact color palette for visualization purposes
palette = pil_mask.getpalette()
```
⚠️ Important Note on Data Loading: We highly recommend using PIL (Pillow) as demonstrated above. Loading these specific masks with standard OpenCV functions (e.g., cv2.imread) without strict grayscale or unchanged flags will often flatten the palette or interpolate values, permanently corrupting the distinct class integer mappings required for training and evaluation.


### Class Label Mapping
The pixel values in the extracted numpy array map to the following categories:
| Pixel Value | RGB Color      | Visual | Class Name             |
|-------------|----------------|--------|------------------------|
| 0           | (0, 0, 0)      | Black  | Background             |
| 1           | (106, 153, 78) | Green  | Good Quality           |
| 2           | (232, 168, 56) | Yellow | Usable Quality         |
| 3           | (193, 69, 69)  | Red    | Bad Quality            |




## Acknowledgement
We build our annotation based on public available datasets and our method on EFIQA, we sincerely thank the authors for their contribution to the field. BRSET and mBRSET is published on PhysioNet.
```
@article{nakayama2024brset,
  title={{BRSET}: a {Brazilian} multilabel ophthalmological dataset of retina fundus photos},
  author={Nakayama, Luis Filipe and Restrepo, David and Matos, Jo{\~a}o and Ribeiro, Lucas Zago and Malerbi, Fernando Korn and Celi, Leo Anthony and Regatieri, Caio Saito},
  journal={PLOS Digital Health},
  volume={3},
  number={7},
  pages={e0000454},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
@article{wu2025mbrset,
  title={A portable retina fundus photos dataset for clinical, demographic, and diabetic retinopathy prediction},
  author={Wu, Chenwei and Restrepo, David and Nakayama, Luis Filipe and Zago Ribeiro, Lucas and Shuai, Zitao and Barboza, Nathan Santos and Sousa, Maria Luiza Vieira and Fitterman, Raul Dias and Pereira, Alexandre Durao Alves and Regatieri, Caio Vinicius Saito and others},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={323},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
@article{Goldberger2000PhysioNet,
  author  = {Goldberger, Ary L. and Amaral, Luis A. N. and Glass, Leon and Hausdorff, Jeffrey M. and Ivanov, Plamen Ch. and Mark, Roger G. and Mietus, Joseph E. and Moody, George B. and Peng, Chung-Kang and Stanley, H. Eugene},
  title   = {PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals},
  journal = {Circulation},
  year    = {2000},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  note    = {RRID:SCR\_007345}
}
@article{BRSET-physio,
  author = {Nakayama, Luis Filipe and Goncalves, Mariana and {Zago Ribeiro}, Lucas and Santos, Helen and Ferraz, Daniel and Malerbi, Fernando and Celi, Leo Anthony and Regatieri, Caio},
  title = {{A Brazilian Multilabel Ophthalmological Dataset (BRSET)}},
  journal = {{PhysioNet}},
  year = {2023},
  month = mar,
  note = {Version 1.0.0},
  doi = {10.13026/xcxw-8198},
}
@article{mBRSET-physio,
  author = {Nakayama, Luis Filipe and {Zago Ribeiro}, Lucas and Restrepo, David and {Santos Barboza}, Nathan and {Dias Fiterman}, Raul and {Vieira Sousa}, Maria luiza and Pereira, Alexandre Durao Alves and Regatieri, Caio and Malerbi, Fernando Korn and Andrade, Rafael},
  title = {{mBRSET, a Mobile Brazilian Retinal Dataset}},
  journal = {{PhysioNet}},
  year = {2024},
  month = jun,
  note = {Version 1.0},
  doi = {10.13026/qxpd-1y65},
}
@inproceedings{
  wang2026efiqa,
  title={{EFIQA: Explainable Fundus Image Quality Assessment via Anatomical Priors}},
  author={Pengwei Wang and Jos{\'e} Morano and Qian Wan and Hrvoje Bogunovi{\'c}},
  booktitle={Medical Imaging with Deep Learning},
  year={2026},
}
```
