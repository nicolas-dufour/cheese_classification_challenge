# Cheese Classification challenge
This codebase allows you to jumpstart the INF473V challenge.
The goal of this channel is to create a cheese classifier without any real training data.
You will need to create your own training data from tools such as Stable Diffusion, SD-XL, etc...

## Instalation

```
conda create -n cheese_challenge python=3.10
conda activate cheese_challenge
pip install -r requirements.txt
```

Download the data from kaggle and copy them in the dataset folder
The data should be organized as follow: ```dataset/val```, ```dataset/test```. then the generated train sets will go to ```dataset/train/your_new_train_set```

## Using this codebase
This codebase is centered around 2 components: generating your training data and training your model.
Both rely on a config management library called hydra. It allow you to have modular code where you can easily swap methods, hparams, etc

### Training

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```

### Generating datasets
You can generate datasets with the following command

```
python generate.py
```

If you want to create a new dataset generator method, write a method that inherits from `data.dataset_generators.base.DatasetGenerator` and create a new config file in `configs/generate/dataset_generator`.
You can then run

```
python generate.py dataset_generator=your_new_generator
```

## Create submition
To create a submition file, you can run 
```
python create_submition.py experiment_name="name_of_the_exp_you_want_to_score" model=config_of_the_exp
```

Make sure to specify the name of the checkpoint you want to score and to have the right model config