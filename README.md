# Fine-Grained Imagination Net for Zero-Shot Learning(FGIN)
This repo contains PyTorch code for paper:

"Fine-Grained Imagination Net for Zero-Shot Learning"

## Requirements
- Python 3.6+
- PyTorch 0.4

## Datasets
### Downloading
You can download the dataset [CUBird and NABird](https://drive.google.com/open?id=1YUcYHgv4HceHOzza8OGzMp092taKAAq1)
Thanks to Elhoseiny et al for their contribution to the dataset.

@inproceedings{Elhoseiny2017Link,
  title={Link the head to the "beak": Zero Shot Learning from Noisy Text Description at Part Precision},
  author={Elhoseiny, Mohamed and Zhu, Yizhe and Han, Zhang and Elgammal, Ahmed and Elhoseiny, Mohamed and Zhu, Yizhe and Han, Zhang and Elgammal, Ahmed and Elhoseiny, Mohamed and Zhu, Yizhe},
  booktitle={Computer Vision & Pattern Recognition},
  year={2017},
}

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
python train_CUB.py --splitmode easy --epoch 3000 --test_number 5
python train_CUB.py --splitmode hard --epoch 3000 --test_number 5
```

#### NABird SCS mode && SCE mode
```
python train_NAB.py --splitmode easy --epoch 20000 --test_number 5
python train_NAB.py --splitmode hard --epoch 20000 --test_number 5
```


 

