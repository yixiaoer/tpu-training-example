# TPU Training Example

## Introduction

This repository showcases an example of fine-tuning a language model using JAX on Google Cloud TPUs. Specifically, it focuses on fine-tuning the T5 model on the WMT14 English-to-French dataset for the machine translation task.

## FAQ

**Q: Why another JAX and TPU example?**

A: (1) JAX is awesome! The more examples with JAX, the better;

(2) This project aims to promote the best practices used herein.

**Q: Why use T5 instead of newer LLMs, such as Llama or Mistral?**

A: For demonstration purposes, smaller models like T5 are chosen due to their lower memory footprint and ease of debugging. This setup allows focusing on the training logic, which remains consistent across model sizes.

## Repository Structure

This repository contains two scripts:

- `train.py`: Fine-tunes the model on a single TPU VM chip without parallelization to provide a basic example of JAX in action.
- `train_shard.py`: Demonstrates distributed training across all TPU chips in a single TPU VM or across all hosts in a TPU Pod.

## Prerequisites

### Configure Your TPU

Set up your TPU VM or TPU Pod with [tpux](https://github.com/yixiaoer/tpux).

### Install Dependencies

Create a virtual environment and install dependencies as follows:

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

For TPU VM, you can directly run the above commands in the terminal.

For TPU Pod, you need to run the above commands on host0 and create the venv in `nfs_share`. Additionally, if you want to run the commands on each host within a TPU Pod, you need to use the `podrun` command from tpux. For instructions on how to use the `podrun` command, please refer to the section below on using the `podrun` command.

### Additional Setup

Update the `WANDB_API_KEY` in the scripts to your own Wandb API key.

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

## Using the `podrun` Command

After configuring your TPU Pod with tpux, if you want to execute the same command across all hosts in the TPU Pod using the `podrun` command from tpux, you can use the following method:

Enter the following command in the terminal:

```sh
podrun -iw
```

After pressing Enter, the `podrun` program will pause and wait for your input. You can copy the command(s) you wish to execute (supporting multiple lines) and paste them into the terminal, press Enter, and then press Ctrl+D to end the input. Afterward, `podrun` will start executing the command across all hosts in the TPU Pod.
