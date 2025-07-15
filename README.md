# Selectivity in MDE

## Instructions

### Download the dataset

The dataset can be downloaded by running
```sh
./download_dataset.sh
```

All the existing configurations are already set to fetch the data from the right directory. Just run the script and the training (as described later).


### Training

This project uses the `uv` package manager for python. If you do have it installed, just run
```sh
uv run train.py configs/your_training_config.yaml
```

If you don't have it or don't want to install it, you can use a docker container as follows:

```sh
docker run -u XXXX:XXXX \
    --rm --ipc host --gpus all -it \
    -w /workspace \
    -e UV_CACHE_DIR=/workspace/.cache \
    -e HOME=/workspace \
    -v $(pwd):/workspace \
    ghcr.io/astral-sh/uv:debian \
    uv run train.py configs/your_training_config.yaml
```

### Evaluation
To run an evaluation you must run the `evaluate.py` script as follows:

```sh
uv run evaluate.py --exp-list exp1,exp2,exp3
```

Or, if using docker:

```sh
docker run -u XXXX:XXXX \
    --rm --ipc host --gpus all -it \
    -w /workspace \
    -e UV_CACHE_DIR=/workspace/.cache \
    -e HOME=/workspace \
    -v $(pwd):/workspace \
    ghcr.io/astral-sh/uv:debian \
    uv run evaluate.py --exp-list exp1,exp2,exp3
```

This command will create a csv file with all the evaluation metrics for every experiment passed as argument.
The names of the experiments correspond to the files in the `config` directory. The checkpoints will be taken from the `experiments` folder
automatically.
