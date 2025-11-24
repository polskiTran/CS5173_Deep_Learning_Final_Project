# Semantic-aware GAN model

## Problem
> “How much do explicit semantic constraints improve sketch→photo realism and structure, over a strong conditional GAN baseline?”  


## How to run notebooks 

### **Option 1** (Recommended)
- Install dependencies: Using [`uv`](https://docs.astral.sh/uv/) ([Rust based python package manager](https://docs.astral.sh/uv/)) *(Recommended)*
```shell
uv sync
```
- Create Jupyter kernel via the command
```shell
uv run ipython kernel install --user --env VIRTUAL_ENV=$(pwd)/.venv --name=dl_proj     
```
- From the jupyter kernel dropdown choose the newly created kernel to run the notebook.

### **Option 2** 
- Install dependencies: Using `requirements.txt` to install dendpencies onto a venv 
```shell
pip install -r requirements.txt
```
- In the jupyter notebook, choose the venv with installed dependcies and run the notebook.




