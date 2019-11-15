# Fine-Grained Generative Adversarial Net(FG-GAN)
This repo contains PyTorch code for paper:

"Fine-Grained Generative Adversarial Net Based Semantic Imagining Model for Zero-Shot Learning"

## Requirements
- Python 3.6+
- PyTorch 0.4

## Datasets
### Downloading
You can download the dataset [CUBird and NABird](https://drive.google.com/open?id=1YUcYHgv4HceHOzza8OGzMp092taKAAq1)

### Preparation
After downloading the datasets above, move them to the `data/` folder, as follows:
```
|-- data
    |-- CUB2011
    |-- NABird
```

## Reproduce results 
#### CUBird SCS mode && SCE mode
```
python train_CUB.py --splitmode easy
python train_CUB.py --splitmode hard
```

#### NABird SCS mode && SCE mode
```
python train_NAB.py --splitmode easy
python train_NAB.py --splitmode hard
```


 
