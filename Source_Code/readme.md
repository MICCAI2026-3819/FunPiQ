# EFIQA-CP

## Install

```bash
pip install -e .
```

## Train

```bash
python -m badish.train --config-name=efiqa
python -m badish.train --config-name=efiqa_cnn
python -m badish.train --config-name=efiqa_cp
```

## Prepare Patch Dataset

```bash
python -m badish.scripts.prepare_patch_dataset \
  --feature-dir /path/to/features \
  --seg-dir /path/to/vuad_masks \
  --img-dir /path/to/images \
  --output /path/to/patch_dataset.npz
```

## Configs

We provide the configurations needed to reproduce the ablation study: EFIQA, EFIQA + CNN adapter and EFIQA + CNN adapter + nnPU loss (EFIQA-CP).

```text
efiqa     = linear adapter + BCE
efiqa_cnn = CNN adapter + BCE
efiqa_cp  = CNN adapter + pixel-level nnPU
```

The configs are in `src/badish/config/`. Update their data paths for your environment.

## Inference

```bash
python -m badish.infer \
  -i /path/to/dino_features \
  -o /path/to/output \
  -m /path/to/model_epochN.pth \
  -c src/badish/config/efiqa_cp.yaml
```


## Reference
For the original EFIQA method, setup, and pipeline details, refer to https://github.com/penway/EFIQA/