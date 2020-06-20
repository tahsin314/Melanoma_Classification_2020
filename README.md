## SIIM-ISIC Melanoma Classification
My scripts for the [SIIM-ISIC Melanoma Classification challenge 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification/).

## Papers
- [Skin lesion classification with ensemble of squeeze-and-excitation networks and semi-supervised learning](https://arxiv.org/abs/1809.02568)


## Resources
- [Margin Focal Loss](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155201)
- [Meta Features](https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet)
- [1st place solution in ISIC 2019 challenge (w/code)](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154683)
- [APTOS Gold Medal Solutions](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108307): ALthough data type is different but it might be helpful.


## Can be useful
- [Deep Metric Learning Solution For MVTec Anomaly Detection Dataset](https://medium.com/analytics-vidhya/spotting-defects-deep-metric-learning-solution-for-mvtec-anomaly-detection-dataset-c77691beb1eb)

## How to run
- Run `git clone https://github.com/tahsin314/Melanoma_Classification_2020`
- Download [this](https://www.kaggle.com/shonenkov/melanoma-merged-external-data-512x512-jpeg) dataset and extract the zip file.
- In the `config.py` file change the `data_dir` variable to your data directory name.
- Run `conda env create -f environment.yml`
- Activate the newly created conda environment.
- Run `train.py`. Change parameters according to your preferences from the `config.py` file before training.


**More resources will be added soon.**