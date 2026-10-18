# Chest X-Ray Multi-Label Diagnosis with DenseNet-121 (PyTorch)

Detects **14 thoracic conditions** (Cardiomegaly, Effusion, Edema, Mass, Pneumothorax, ...) from a single frontal chest X-ray by fine-tuning an ImageNet-pretrained **DenseNet-121** on the **NIH ChestX-ray14** dataset, with **Grad-CAM** heatmaps showing which region drove each prediction.

## Results

_Training run in progress - per-class test ROC-AUC, ROC curves and Grad-CAM examples will be added here._

## Approach

| Step | What I did | Why |
|---|---|---|
| Data split | Official NIH test list + patient-grouped train/val split, with an automated leakage check | The same patient has many X-rays; a random split leaks and inflates AUC |
| Preprocessing | Resize 1024 to 256 once and cache in RAM; random-resized crop, rotation, brightness/contrast jitter; **no horizontal flip** | Fast epochs; flipping would put the heart on the wrong side |
| Model | DenseNet-121 (ImageNet weights) + 14-unit sigmoid head | Multi-label: one X-ray can show several conditions |
| Class imbalance | BCE with per-class `pos_weight = neg/pos`, capped at 50 | Hernia is ~0.2% of images; uncapped weights destabilise training |
| Training | AdamW with 10x higher LR for the head than the backbone, OneCycle schedule, mixed precision, early stopping on validation macro-AUC | Stable fine-tuning on a free Kaggle GPU |
| Explainability | Grad-CAM on the last dense block | Checks the model looks at plausible anatomy, not text markers or devices |

## Run it

1. Open `chest_xray_densenet121_pytorch.ipynb` on **Kaggle**, add the dataset **`nih-chest-xrays/data`**, set the accelerator to **GPU** and turn **Internet** on.
2. Run with `SMOKE_TEST = True` first (~3 min), then `False` for the full run.
3. Outputs (model weights, metrics, plots) are written to `/kaggle/working/outputs`.

## Limitations

- NIH labels were text-mined from radiology reports and are noisy, which caps achievable AUC.
- This is a learning project, not a medical device.

## References

- Wang et al., *ChestX-ray8*, CVPR 2017 - dataset
- Rajpurkar et al., *CheXNet*, 2017 - DenseNet-121 baseline on this dataset
- Selvaraju et al., *Grad-CAM*, ICCV 2017
