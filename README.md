# Coupled-hypersphere-based Feature Adapatation for Target-Oriented Anomaly Localization

## Getting Started

Install packages with:

```
$ pip install -r requirements.txt
```

## Data

Prepare industrial image as:

``` 
train data:
    dataset_path/class_name/train/good/any_filename.png
    [...]

test data:
    dataset_path/class_name/test/good/any_filename.png
    [...]

    dataset_path/class_name/test/defect_type/any_filename.png
``` 

## How to train

### Example
```
python trainer_cfa.py --class_name all --data_path [/data/path/] --cnn wrn50_2 --wild False
```

## Reference
[1] https://github.com/byungjae89/SPADE-pytorch

[2] https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

[3] https://github.com/pytorch/vision/tree/main/torchvision/models

[4] https://github.com/lukasruff/Deep-SVDD-PyTorch