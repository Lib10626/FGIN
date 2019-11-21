# Fine-Grained Imagination Net for Zero-Shot Learning(FGIN)
This repo contains PyTorch code for paper:

"Fine-Grained Imagination Net for Zero-Shot Learning"

## Requirements
- Python 3.6+
- PyTorch 0.4

## Datasets
### Downloading
You can download the dataset [CUBird and NABird](https://drive.google.com/drive/folders/1NS_e_r9-nTWlGhNcpKpY5X7Mwo3oVBL3?usp=sharing)

Thanks to the contribution of this paper [Link the head to the "beak": Zero Shot Learning from Noisy Text Description at Part Precision](https://arxiv.org/pdf/1709.01148.pdf) to the dataset.

### Preparation
After downloading the datasets above, move them to the `data/` folder, as follows:
```
|-- data
    |-- CUB2011
    |-- NABird
```

## Reproduce results 
#### CUBird with SCS-setting(Easy Mode)
```
python train_CUB.py --splitmode easy --epoch 3000 --test_number 5
```

#### CUBird with SCE-setting(Hard Mode)
```
python train_CUB.py --splitmode hard --epoch 3000 --test_number 5
```

#### NABird with SCS-setting(Easy Mode)
```
python train_NAB.py --splitmode easy --epoch 20000 --test_number 5
```

#### NABird with SCE-setting(Hard Mode)
```
python train_NAB.py --splitmode hard --epoch 20000 --test_number 5
```


 

