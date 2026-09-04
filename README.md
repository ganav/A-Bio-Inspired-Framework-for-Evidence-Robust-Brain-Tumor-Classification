# Brain MRI Paper Source Code

This package replaces the previous chest-X-ray implementation with a version
for the **Brain Tumor MRI Dataset** by Masoud Nickparvar on Kaggle.

## 1. Expected MRI dataset layout

Keep the Kaggle folder structure:

```text
Brain-Tumor-MRI-Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The code makes a **stratified validation split from the Training folder**
(default: 10%). The original Testing folder is kept for final evaluation.

## 2. Brain-region masks

The Kaggle dataset does not contain paired brain-region masks. The proposed
method therefore requires a separate mask root:

```text
brain_masks/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Each mask must have the same filename stem as the corresponding MRI.

Example:

```text
Brain-Tumor-MRI-Dataset/Training/glioma/Tr-gl_001.jpg
brain_masks/Training/glioma/Tr-gl_001.png
```

**White = brain region; black = non-brain region.**

Do not call these masks "ground truth" in the manuscript unless they are
genuinely verified ground-truth annotations.

### If you already removed the non-brain region

If you have a mirrored set of cleaned images in which the non-brain area is
black, generate the binary masks with:

```bash
python make_masks_from_cleaned.py \
  --cleaned-root cleaned_brain_images \
  --mask-root brain_masks
```

This helper converts your already-cleaned images into masks. It is not a
medical segmentation model.

## 3. Install

```bash
pip install -r requirements.txt
```

## 4. Proposed DenseNet-121 experiment

```bash
python train_brain_mri_framework.py \
  --data-root Brain-Tumor-MRI-Dataset \
  --mask-root brain_masks \
  --output-dir outputs/densenet121_proposed \
  --backbone DenseNet-121 \
  --method proposed \
  --image-size 224 \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --lambda-cam 1.0 \
  --lambda-confidence 1.0 \
  --lambda-dominance 0.5 \
  --blur-kernel 11 \
  --tile-grid 4
```

## 5. Baseline DenseNet-121 experiment

```bash
python train_brain_mri_framework.py \
  --data-root Brain-Tumor-MRI-Dataset \
  --mask-root brain_masks \
  --output-dir outputs/densenet121_baseline \
  --backbone DenseNet-121 \
  --method baseline \
  --image-size 224 \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 0.001
```

For the other backbones use:

```text
--backbone MobileNet-v1
--backbone EfficientNet-B0
```

## 6. What the proposed code implements

The proposed training objective contains four terms:

1. **Classification loss**
2. **CAM attention loss**
3. **Confidence-inconsistency loss**
4. **Single-evidence-dominance loss**

The three evidence reductions are:

- Gaussian blur, default **11 x 11**, for texture reduction
- **4 x 4 tile shuffle** (16 tiles) for shape disruption
- **50% intensity reduction** for pixel-intensity reduction

The same three perturbed logits are reused for both confidence inconsistency
and evidence dominance.

## 7. Outputs

Each experiment directory contains:

```text
configuration.json
epoch_log.csv
best_model.weights.h5
brain_mri_classifier.keras
brain_mri_classifier.h5
test_metrics.json
confusion_matrix.csv
class_metrics.csv
classification_report.txt
cam_overlays/
```

`class_metrics.csv` contains class-wise **PPV, TPR, F1, and support**.

The exported classifier requires only the original MRI at inference time.

## 8. Statistical comparison

Do not invent a p-value or Cohen's d from a single baseline/proposed pair.
After repeated paired runs, create:

```text
runs.csv
run,baseline_f1,proposed_f1
1,0.896,0.928
2,...
3,...
```

Then run:

```bash
python compare_runs.py runs.csv
```

It reports a paired t-test p-value and paired Cohen's dz.

## 9. Important manuscript update

The earlier X-ray code expected COVID, Lung Opacity, Normal, and Viral
Pneumonia folders plus lung masks. This MRI version instead uses:

- Glioma
- Meningioma
- Pituitary
- No Tumor

and uses brain-region masks.
