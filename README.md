# Chest X-Ray Multi-Label Diagnosis with DenseNet-121 (PyTorch)

Detects **14 thoracic conditions** (Cardiomegaly, Effusion, Edema, Mass, Pneumothorax, ...) from a single frontal chest X-ray by fine-tuning an ImageNet-pretrained **DenseNet-121** on the **NIH ChestX-ray14** dataset, with **Grad-CAM** heatmaps showing which region drove each prediction.

## Results

Fine-tuned on **30,000 training images**, early-stopped on validation macro-AUC (best 0.828 at epoch 5), and evaluated on **10,000 held-out images** from the official NIH test split. No patient appears in more than one split.

**Test macro ROC-AUC: 0.775** (14 classes)

| Condition | Test AUC | Test positives |
|---|---|---|
| Emphysema | 0.864 | 406 |
| Cardiomegaly | 0.847 | 444 |
| Hernia | 0.845 | 31 |
| Pneumothorax | 0.829 | 1047 |
| Edema | 0.815 | 350 |
| Effusion | 0.804 | 1866 |
| Fibrosis | 0.800 | 182 |
| Mass | 0.783 | 704 |
| Atelectasis | 0.747 | 1300 |
| Pleural thickening | 0.724 | 439 |
| Nodule | 0.721 | 655 |
| Consolidation | 0.717 | 724 |
| Infiltration | 0.688 | 2373 |
| Pneumonia | 0.668 | 220 |
| **Macro average** | **0.775** | |

Training: 8 epochs on one Kaggle T4 GPU (~2.7 min/epoch with mixed precision). Validation AUC peaked at epoch 5 and the best checkpoint was used for testing. Test AUC (0.775) is lower than validation AUC (0.828); scaling up to all ~78K training images is the next step.

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
