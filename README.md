# Selectivity in MDE

## Instructions

### Download the dataset

The dataset can be downloaded by running
```sh
./download_dataset.sh
```

**Please note**: the config files are already set to pick the dataset from the output folder of the above script. By not changing the path in the
configurations, training script will work correctly.

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
uv run evaluate.py --exp-list exp1,exp3,exp10,exp11
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
    uv run evaluate.py --exp-list exp1,exp3,exp10,exp11
```
By specifying the experiment name, the script will automatically pick the best checkpoint (in terms of validation loss)
from the `experiments` folder. The repository includes the best checkpoints for experiments 1, 3, 10, 11.
In order to select a different one, you'll need to run the training specifyint the corresponding configuration file.

This command will create a csv file with all the evaluation metrics for every experiment passed as argument.
The names of the experiments correspond to the files in the `config` directory. The checkpoints will be taken from the `experiments` folder
automatically.
