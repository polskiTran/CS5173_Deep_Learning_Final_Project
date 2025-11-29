# Semantic-aware GAN model

## Problem
> “How much do explicit semantic constraints improve sketch→photo realism and structure, over a strong cycle GAN baseline?” - An implementation of [2024 DLP GAN paper](https://arxiv.org/abs/2403.03456) using Abstract Chinese Landscape Painting and Photo of Landscape 

## Architecture Overview (Based on [DLP GAN](https://arxiv.org/abs/2403.03456))
> The core framework relies on an asymmetric mapping strategy to handle the information imbalance between the two domains: Domain $X$ (Paintings/Abstract) and Domain $Y$ (Photos/Dense Information).
### Generator
**Generator Strict ($G:X→Y)$:**
- Purpose: Translating abstract paintings to detailed photos. This requires learning the color distribution of landscape targets in the photos and recreate in the paintings. 
- Architecture: Uses a Dense Fusion Block mechanism. Unlike standard residual connections, this block connects layers densely to enhance feature propagation and reuse, allowing the network to preserve semantic content while generating complex realistic textures.
- Initialization: Gaussian initialization (μ=0,σ=0.02) is applied to ensure stable starting gradients.

**Generator Relaxed ($F:Y→X$):**
- Purpose: Translating detailed photos to abstract paintings.
- Architecture: Utilizes a standard ResNet-based architecture.
- Modification: The current implementation utilizes 9 Residual Blocks (increased from the typical 6 for smaller images). This increases the capacity of the network to handle the 512x512 resolution of the input data.
### Discriminator
- The system employs two PatchGAN discriminators ($D_X$​,$D_Y$​). These classify 70×70 overlapping image patches as real or fake, focusing on high-frequency structural correctness rather than global image context.

## Objective Functions
- In DLP GAN paper:
```math
\begin{align}
\min_{G,F}\;\max_{D_X,D_Y}\; 
L_{\text{Total}}(G, F, D_X, D_Y)
&= \lambda_{\text{GAN}} L_{\text{GAN}}
 + \lambda_{\text{Dual}} L_{\text{Dual}}
 + \lambda_{\text{id}} L_{\text{id}} \\
&= \lambda_{\text{GAN}}
   \left( 
     L_{\text{LSGAN}}(G, D_Y)
     + L_{\text{LSGAN}}(F, D_X)
   \right) \\
&\quad + \lambda_{\text{Dual}}
   \left(
      L_{\text{Feature}}(G, F)
      + \mu L_{\text{Semantic}}(G, F)
   \right) \\
&\quad + \lambda_{\text{id}} L_{\text{id}}(G, F)
\end{align}
```
- The training objective is a weighted sum of four loss components:
    - Adversarial Loss ($L_{\text{GAN}}$​): Utilizes Least Squares GAN (LSGAN) loss rather than standard Cross-Entropy. This provides smoother gradients and penalizes samples based on their distance from the decision boundary, reducing vanishing gradient problems.
    - Dual-Consistency Loss ($L_\text{Dual}$): Designed to balance realism and abstraction by enforcing consistency at two levels:
        - Feature Consistency ($L_\text{Feature}$): Computes the L1​ distance between VGG-16 (relu3_3) feature maps of the real and reconstructed images.
        - Semantic Consistency ($L_\text{Semantic}$)): Computes the L1​ distance between edge maps of the real and generated images.
        > Note: While the original paper uses DexiNed for semantic edge extraction, the current implementation utilizes a differentiable Sobel Filter approximation.
    - Identity Loss ($L_\text{id}$): Standard L1​ loss between input and output when the target domain image is fed into the generator (e.g., $G(y)≈y$). Used to preserve color composition.

## Training Config
- The current setup diverges from the original paper's hyperparameters to match with the dataset 

- Hyperparameters
    - Learning Rate: 0.0002 (Adam Optimizer, β1​=0.5,β2​=0.999).
    - Batch Size: 2 (Instance Normalization standard).
    - Image Resolution: 512×512.

- Weighting Strategy: The loss weights (λ) have been tuned to prioritize adversarial realism over strict content matching:
    - $λ_\text{GAN}​=4$ (Original: 1).
        - Rationale: Significantly increased to force the Generator to prioritize "fooling" the discriminator. This combats the issue where the generator simply outputs the input image to satisfy reconstruction loss.

    - $λ_\text{Dual}​=8$ (Original: 10).
        - Rationale: Slightly reduced. High dual consistency acts as a rigid constraint; lowering it allows the generator more freedom to "hallucinate" realistic textures that don't align perfectly with the sketch lines.

    - $λ_\text{ID}=3$ (Original: 5).
        - Rationale: Reduced. High identity loss forces the model to mimic the input's color histogram. Since sketches/paintings are often monochromatic, high Lid​ prevents the model from learning the vibrant colors of the photo domain.

## Directory Index
- `Data/`
    - `xue2020_dataset` - 2,000+ Abstract Chinese Paintings (512x512) collected thanks to [xue, 2020](https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset)
    - `data_processing.ipynb` - Notebook for data downloading and processing (only applies to Lanscape dataset)
    - `data_overview.md` - Overview of experimented dataset
    - `hidden_test` - Some cherry picked hidden test painting for Painting2Photo
- `results_[...]\` - folders of experimented model results and demo. Check `results_CANDIDATE]` for latest models weights
    - `checkpoints` - contains model weights
    - `graghs` - training graphs
    - `images` - training images validation
- `model_training.ipynb` - Notebook for model init and training
- `model_training.py` - Generated python script mirroring `model_training.ipynb` (refer to "Guide for long remote access train (>12h) on department hardware" down below). 
- `model_evaluation` - Inference for trained model - GO HERE FOR GENERATING IMAGES FROM TEST DATASET
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

### Guide for long remote access train (>12h) on department hardware
- Download `uv` using the command
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Use tmux to create a persistant terminal session so the training keep runing even after closing remote access.
```shell
tmux
``` 
- To export the `.ipynb` file to `.py` script run the command
```shell
jupyter nbconvert --to script model_training.ipynb
```
- To run the training script run the command
```shell
uv run model_training.py
```
- Detach the current session so it continue running in background via `Ctrl + b d` (hitting ctrl key and b key at the same time then d key)
- You are now good to close the remote access session
- To go back to the running training session use command
```shell
tmux attach
```


