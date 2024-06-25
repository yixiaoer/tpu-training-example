# TPU Training Example

## Introduction

This repository is an example of training a language model on Google Cloud TPU using JAX. Specifically, it fine-tunes a T5 model on the machine translation task, using wmt14 English to French dataset.

There are two scripts in this repository, namely `train.py` and `train_shard.py`.

The first script, `train.py`, fine-tunes the model on a single TPU chip, without any parallelisation. The purpose of this script is to give a basic example of how JAX works.

The second script, `train_shard.py` fine-tunes the model on all chips of a TPU VM or all hosts in a TPU Pod. The purpose of this script is to demonstrate how to perform distributed training using JAX. 

The second script should support fine-tuning on multiple devices, which includes both (1) all TPU chips on a single host and (2) a TPU Pod. ~~However, due to a bug of JAX (https://github.com/google/jax/issues/22070), only (1) is currently supported.~~ <!-- This involves running the script on all hosts, using the `podrun -iw` command. -->

## Prerequisites

### Config your TPU

After creating a TPU VM or a TPU Pod, the first step is to configure it using [tpux](https://github.com/yixiaoer/tpux).

### Install dependencies

Then, create a venv and install dependencies:

```sh
# For TPU Pod, you should git clone in the `nfs_share` after using tpux to config
git clone https://github.com/yixiaoer/tpu-training-example.git
cd tpu-training-example
python3.12 -m venv ~/venv
. ~/venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[cpu]"
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

On TPU VM, you can directly run the above commands in the terminal. 

On TPU Pod, you run the above command on host0 to create a venv in `nfs_shared`, and in [training part](https://github.com/yixiaoer/tpu-training-example#training) you will see how to use this one venv to run across hosts.

On TPU Pod, if you want to run the above commands on every host, the way to do this is to utilise the `podrun` command from tpux. Run the following command:

```sh
podrun -iw
```

The `podrun` program will pause and wait for your input. You can copy the above commands and paste them into the terminal, press enter, then press Ctrl+D to terminate the input. `podrun` will then start to run the commands on all TPU hosts.

### Others

You also need to set the `WANDB_API_KEY` in the proper location of the script to add your WANDB API key.

## Training

* For TPU VM (using a single device), run the following command:

    ```sh
    # in `tpu-training-example` directory
    . ~/venv/bin/activate
    python train.py
    ```

    Go to [Wandb page](https://wandb.ai/yiixiaoer/training-t5/runs/agtpl7zk) to see the loss decrease.

* For TPU VM (using all TPU chips), run the following command:

    ```sh
    # in `tpu-training-example` directory
    . ~/venv/bin/activate
    python train_shard.py
    ```

    Go to [Wandb page](https://wandb.ai/yiixiaoer/training-t5/runs/8osdw903) to see the loss decrease, while `learning_rate = 1e-5`.

* For TPU Pod (using multi hosts), run the following command on your host0:

    ```sh
    # in `tpu-training-example` directory on host0 
    # only create a venv on host0 in `~/nfs_share/tpu-training-example`

    # Directly run podrun, and it will wait for your input
    podrun -iw
    # enter the follwing line for `podrun`, then press enter
    venv/bin/python train_shard.py
    ```

    Or you can also stil run with `echo` and `|`:

    ```sh
    # in `tpu-training-example` directory on host0
    # only create a venv on host0 in `~/nfs_share/tpu-training-example`

    echo venv/bin/python train_shard.py | podrun -iw
    ```
