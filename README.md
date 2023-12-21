# Bachelor Thesis (title is TBA)

## How to compile
0. Make sure to clone this repo with submodules:

   ```shell
   git clone git@github.com:schweikart/bachelor-thesis.git --recursive
   ```
1. Install the full TexLive distribution:
  
   ```shell
   sudo apt-get install texlive-full
   ```
2. Generate document through LaTeXmk:

   ```shell
   latexmk
   ```
3. Done!
   You can find the document at [`./thesis.pdf`](./thesis.pdf).

## Evaluation Server
This section describes the server installation used to run the evaluation.

* Server: [Hetzner CAX41 ARM64 cloud server](https://www.hetzner.com/cloud) with 16 shared vCPUs, 32GB of RAM, and 320GB of disk space.
* Software:
  * Operating system: Ubuntu 22.04
  * Package installation:
    ```
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python-is-python3 python3-pip python3-venv
    ```
  * Virtual environment setup:
    ```
    # in repository folder
    python3 -m venv .venv
    source ./.venv/bin/activate
    pip install -r ./code/requirements.txt
    ```
  * Start Jupyter server:
    ```
    jupyter notebook
    # note: set up SSH tunnel before connecting to server
    # ssh -L 8888:localhost:8888 root@23.88.96.193
    ```
  * Run evaluation (use `tmux` to keep running in the background):
    ```
    cd crotosolve/code
    jupyter nbconvert --to script dataset_generation.ipynb
    ipython dataset_generation.py
    ```
    > **⚠️ Important:**
    > Make sure to commit generated evaluation data to the repository!



