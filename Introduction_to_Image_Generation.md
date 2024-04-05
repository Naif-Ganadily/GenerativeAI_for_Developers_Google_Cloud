[Link to the course](https://www.cloudskillsboost.google/paths/183/course_templates/541)

## Image Generation Model Families
**Variational Autoencoders - VAEs**
- Encode images to a compressed size, then decode back to the original size, while learning the distribution of the data
**Generative Adversarial Models - GANs**
- Pit two neural networks against each other
**Autoregressive Models**
- Generate images by treating an image as a sequence of pixels

## Diffusion Model: New trend of generative model

## Diffusion Model: Use Cases
**Unconditioned Generation**
- Human face synthesis
- Super-resolution
**Conditioned Generation**
- Text-to-image
- Image-inpainting
- Text-guided image-to-image

## Diffusion Model: What is it?
- The essential idea is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data. 

## Denoising Diffusion Probabilistic Models (DDPM)
$p_{\theta}(x_{t-1} | x_t)$

$q(x_t | x_{t-1})$

- In each step, add gaussian noise to images. -> $q(x_t | x_{t-1})$
- T: Time step diffusion process (T=1000 in DDPM).
- In each step, remove gaussian noise from. This need to be learned -> $p_{\theta}(x_{t-1} | x_t)$

## DDPM Training
1. Start with an original data point `X_0` (the clear image).
2. Apply noise over time `t` (timestep) to create a noised data point `X_t`.
3. The `Denoising Model` (U-Net + Attention) takes `X_t` as input.
4. The model predicts the noise that was added to `X_0` to obtain `X_t`.
5. The `Predicted Noise` is then compared to the actual noise using a `Loss` function (Pixel-wise MSE).
6. The parameters of the Denoising Model are updated based on the loss to improve noise prediction.

## DDPM Generation Process

1. Start with `Noise` - a random noise image.
2. Pass the noise through a `Denoising Model` to predict the noise.
3. Subtract the `Predicted Noise` from the noised image.
4. Pass the result through the `Denoising Model` again.
5. Repeat the denoise-predict-subtract process for multiple timesteps `...`.
6. Finally, obtain `X_0`, the denoised image representing the generated data.

## Recent Advancements
- Many improvements have been made already!
- - Faster generation
- - Conditioned models
- - CLIP
- Text-to-image models combine the power of diffusion models with LLMs. 

Like Imagen from Google Research, which uses a diffusion model to generate images from text prompts.

