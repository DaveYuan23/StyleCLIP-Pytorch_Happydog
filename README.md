# StyleCLIP-Pytorch_Happydog

## Results Showcase
### Images
![Happy Dog Image 1](final_photo/final_result.jpg)
![Happy Dog Image 2](final_photo/final_result2.jpg)
![Happy Dog Image 3](final_photo/final_result3.jpg)

### Videos
video 1

https://github.com/user-attachments/assets/503fa6ce-b7ea-4841-b23a-087aabfd45d1

video 2

https://github.com/user-attachments/assets/7c1ee0cc-57ff-4040-83fa-8a52a59627dd

## Latent Optimization Step

A simple approach for leveraging CLIP to guide image manipulation is through direct latent code optimization. This method involves three essential components:

1. **Requirements**:
   - A pre-trained StyleGAN model.
   - A source latent code (usually generated from random noise \( z \) by the mapper from the generator; we can also perform image inversion using e4e to edit the image of our choice).
   - A pre-trained CLIP model.

2. **Loss Function**:
![Optimization Problem](final_photo/loss_function.png)
   The loss function consists of three parts:
   - **CLIP Loss** $$\( D_{clip} \)$$: This calculates the cosine distance between the CLIP embeddings of the text and image arguments, where $$G$$ is a pre-trained StyleGAN generator and $$t$$ is the text prompt.
   - **L2 Norm**: This part calculates the L2 distance between the source latent code $$w_s$$ and the target latent code $$w$$.
   - **Identity Loss**: Ensures that the identity of the image remains unchanged while allowing modifications to other visual features (e.g., hairstyle, expression, presence of glasses, etc.). The identity loss is calculated using a pre-trained ArcFace network for face recognition.

3. **Finding the Optimized $$w$$**:
   We find the optimized $$w$$ by solving the optimization problem through gradient descent. The gradient of the objective function is backpropagated while freezing the pre-trained StyleGAN and CLIP models. Typically, using an epoch count of 150 to 250 will yield decent results. The $\lambda 2$ parameter usually ranges from 0.02 to 0.06, depending on the extent to which you want to change your photo. The $\lambda_ID$ parameter is only appied when editing human faces.

## Getting Started

### Install Pre-trained CLIP Model
To install the pre-trained CLIP model, run the following command:

```bash
pip install git+https://github.com/openai/CLIP.git
```
### Install Ninja
To install Ninja, execute the following commands:
```bash
!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
!sudo unzip ninja-linux.zip -d /usr/local/bin/
!sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```
### Get Pre-trained StyleGAN
Clone the StyleGAN repository:
```bash
!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
%cd stylegan2-ada-pytorch
```

## Problems Encountered During This Project

1. **Using Pre-trained StyleGAN Model**: Install Ninja and modify the code in `torch.utils`.
2. **Using Mac**: Unable to train the network due to an unknown reason. The error message was: Expected scalar_type == ScalarType::Float || inputTensor.scalar_type() == ScalarType::Int || scalar_type == ScalarType::Bool to be true, but got false.
3. **CLIP Installation**: Install directly from OpenAI's CLIP to avoid errors. Do not use `pip install CLIP`.
4. **Loss Function Parameters**: Adjust the parameters in the loss function to achieve the ideal result.

## References

1. [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) StyleGAN2-ADA — Official PyTorch implementation

2. [encoder for editing](https://github.com/omertov/encoder4editing) e4e — encoder for editing

3. [CLIP](https://github.com/openai/CLIP)
