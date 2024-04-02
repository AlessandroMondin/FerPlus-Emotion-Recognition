# FERPlus with Pytorch Lightning

This repository allows to train model on the FERPlus dataset, in order to create models able to detect human emotions.

## Setup FERPlus dataset

- Download the FERPlus dataset by following the simple instructions present in the official repository: [link](https://github.com/microsoft/FERPlus)
- After creating the dataset, you will end up with a folder containing with the following structure: <br>

```
fer2013
├── FER2013Train
│   ├── fer0000000.png
│   ├── fer0000001.png
│   ├── fer0000002.png
│   ├── ...
│   ├── fer0028637.png
│   └── label.csv
├── FER2013Valid
│   ├── fer0028638.png
│   ├── fer0028639.png
│   ├── fer0028640.png
│   ├── ...
│   ├── fer0032219.png
│   └── label.csv
└── FER2013Test
    ├── fer0035780.png
    ├── fer0035781.png
    ├── fer0035782.png
    ├── ...
    ├── fer0035801.png
    └── label.csv
```

## Setup Python environment:

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## How to launch a training process

At this point, in order to start training a model you need to specify the training configurations in `config.py` and eventually add a your custom models under the `models` folder. Bear in mind, if you add custom models, you'll need to inport them in `train.py`.

Once you have specified these two configurations you can launch training by running `python train.py`
