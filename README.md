# SSM
This is our Tensorflow implementation for SSM.

This project is based on [NeuRec](https://github.com/wubinzzu/NeuRec/). Thanks to the contributors.

## Environment Requirement

The code runs well under python 3.7.7. The required packages are as follows:

- Tensorflow-gpu == 1.15.0
- numpy == 1.19.1
- scipy == 1.5.2
- pandas == 1.1.1
- cython == 0.29.21

## Quick Start

Firstly, download this repository and unpack the downloaded source to a suitable location.

Secondly, go to '*./NeuRec*' and compline the evaluator of cpp implementation with the following command line:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

Thirdly, specify dataset and recommender in configuration file *NeuRec.properties*.

Finally, run [main.py](./main.py) in IDE or with command line:

### Gowalla dataset
```bash
python main.py --recommender=SSM --data.input.dataset=gowalla --n_layers=1 --temp=0.12 --reg=1e-5
```

### Yelp2018 dataset
```bash
python main.py --recommender=SSM --data.input.dataset=yelp2018 --n_layers=1 --temp=0.14 --reg=1e-5
```

### Amazon-Book dataset
```bash
python main.py --recommender=SSM --data.input.dataset=amazon-book --n_layers=3 --temp=0.1 --reg=1e-1
```

### Alibaba-iFashion dataset
```bash
python main.py --recommender=SSM --data.input.dataset=ifashion --n_layers=1 --temp=0.22 --reg=1e-3
```