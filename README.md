# StyleCLIP-Pytorch_Happydog

## Results Showcase

### Images
![Happy Dog Image 1](path/to/image1.jpg)
![Happy Dog Image 2](path/to/image2.jpg)
![Happy Dog Image 3](path/to/image3.jpg)

### Videos
[![Happy Dogs Video](path/to/thumbnail.jpg)](path/to/video.mp4)  
*Click on the image above to watch the video showcasing happy dogs!*

## Latent Optimization Step

A simple approach for leveraging CLIP to guide image manipulation is through direct latent code optimization. This method involves three essential components:

1. **Requirements**:
   - A pre-trained StyleGAN model.
   - A source latent code (usually generated from random noise \( z \) by the mapper from the generator; we can also perform image inversion using e4e to edit the image of our choice).
   - A pre-trained CLIP model.

2. **Loss Function**:
   $$
   \text{arg min} \ D_{\text{CLIP}}(G(w), t) + \lambda_{L2} \| w - w_s \|_2 + \lambda_{ID} L_{ID}(w), 
   \quad w \in W^+
   $$
   The loss function consists of three parts:
   - **CLIP Loss** \( D_{clip} \): This calculates the cosine distance between the CLIP embeddings of the text and image arguments, where \( G \) is a pre-trained StyleGAN generator and \( t \) is the text prompt.
   - **L2 Norm**: This part calculates the L2 distance between the source latent code \( w_s \) and the target latent code \( w \).
   - **Identity Loss**: Ensures that the identity of the image remains unchanged while allowing modifications to other visual features (e.g., hairstyle, expression, presence of glasses, etc.). The identity loss is calculated using a pre-trained ArcFace network for face recognition.

   ![Loss Function Diagram](path/to/loss_function_diagram.jpg)

4. **Finding the Optimized \( w \)**:
   We find the optimized \( w \) by solving the optimization problem through gradient descent. The gradient of the objective function is backpropagated while freezing the pre-trained StyleGAN and CLIP models.

## Getting Started

### Install Pre-trained CLIP Model
To install the pre-trained CLIP model, run the following command:

```bash
pip install git+https://github.com/openai/CLIP.git
