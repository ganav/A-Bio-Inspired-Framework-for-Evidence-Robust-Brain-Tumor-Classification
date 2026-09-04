# Brain MRI Paper Source Code

Expected MRI dataset layout

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

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download

Dataset Description
The dataset contains 7,200 human brain MRI images categorized into four classes:

Glioma
Meningioma
Pituitary tumor
No tumor
The dataset is structured into training and testing sets with balanced class distributions.

This dataset is a curated combination of the following publicly available datasets:

figshare brain MRI dataset
SARTAJ dataset
Br35H dataset
Images in the no tumor class were sourced from the Br35H dataset.

During dataset preparation, inconsistencies were observed in the glioma class of the SARTAJ dataset. To improve label reliability, those images were removed and replaced with verified glioma images from the figshare dataset.

Version 2 Updates
Removed duplicate images
Balanced classes (1400 training / 400 testing per class)
Eliminated overlap between training and testing sets to prevent data leakage

Note
Image sizes vary across the dataset.
Preprocessing steps such as resizing, normalization, and margin removal (sample code) are recommended before training models.
Proper preprocessing can significantly improve model performance.
If you find this dataset useful for your research, please consider upvoting it. Feedback and suggestions are welcome. ❤️
