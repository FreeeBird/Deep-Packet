# Deep Packet

Details in blog post: https://blog.munhou.com/2020/04/06/Pytorch-Implementation-of-Deep-Packet-A-Novel-Approach-For-Encrypted-Tra%EF%AC%83c-Classi%EF%AC%81cation-Using-Deep-Learning/

## How to Use

* Clone the project
* Download the train and test set I created at [here](https://drive.google.com/file/d/1_O2LPs3RixaErigJ_WL1Ecq83VXCXptq/view?usp=sharing), or download the [full dataset](https://www.unb.ca/cic/datasets/vpn.html) if you want to process the data from scratch.

## Data Pre-processing

```bash
python preprocessing.py -s /path/to/CompletePcap/ -t processed_data
```

## Create Train and Test

```bash
python create_train_test_set.py -s processed_data -t train_test_data
```

## Train Model

Application Classification

```bash
python train_cnn.py -d train_test_data/application_classification/train.parquet -m model/application_classification.cnn.model -t app
```

Traffic Classification

```bash
python train_cnn.py -d train_test_data/traffic_classification/train.parquet -m model/traffic_classification.cnn.model -t traffic
```

## Evaluation Result
### Application Classification
![](https://blog.munhou.com/images/deep-packet/cnn_app_classification.png)

### Traffic Classification
![](https://blog.munhou.com/images/deep-packet/cnn_traffic_classification.png)

## Model Files

Download the pre-trained models [here](https://drive.google.com/file/d/1UgSqcN5SG5hqC2imlYu6bB2f9jD1iiu8/view?usp=sharing).
