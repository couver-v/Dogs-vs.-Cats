# Dogs vs. Cats

### Kaggle Challenge : Dogs vs. Cats
https://www.kaggle.com/c/dogs-vs-cats

## Install dependances

### Use conda

install anaconda or miniconda:

    https://conda.io/docs/user-guide/install/index.html

create virtual environment with :

    conda env create -f env-conda.yml
    source activate tf

### Install on you own env

    pip install tensorflow tqdm Pillow

## Usage

### Download data

Download the train.zip file located in the data tab of the Kaggle challenge and unzip it at the root of the repository

### Preprocessing
    python3 prepare_data.py
    
### Compare images before and after preprocessing
    python3 show_image.py

### Training
    python3 dogsvscats.py

![Cats vs Dogs](http://mimibuzz.com/plugin/buzz/view/resource/public/img/image/12109/original.jpg)
