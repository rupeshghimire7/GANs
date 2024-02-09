# GANs
<p>
A Generative Adversarial Network (GAN) consists of two neural networks: a generator and a discriminator. The generator aims to produce realistic data samples from random noise, while the discriminator learns to distinguish between real and fake samples. During training, the generator and discriminator play a minimax game, where the generator tries to fool the discriminator by generating realistic samples, and the discriminator tries to correctly classify real and fake samples. This adversarial training process iterates until the generator produces realistic samples indistinguishable from real data.
</p>
<p>
This repository contains implementations of various Generative Adversarial Networks (GANs) models in PyTorch. Each GAN model is implemented in a separate directory along with Jupyter Notebook files for experimentation and learning.
</p>

## Overview of Implemented GAN Models

### 1. Simple GAN:
<p> GAN is a generative model consisting of two neural networks, a generator and a discriminator, trained in a two-player minimax game setting. The generator learns to generate realistic data samples, while the discriminator learns to distinguish between real and fake data. 
<br>
This uses the Linear model and Leaky ReLU activation in the Discriminator and Generator's inner layers. It uses Sigmoid and Tanh on the last layer of the Discriminator and Generator respectively.
</p>

### 2. DCGAN:
<p>
DCGAN is an extension of GAN that uses deep convolutional networks in both the generator and discriminator architectures. It achieves better stability and quality in image generation tasks compared to simple GAN.
<br>
Incorporates deep convolutional networks for both generator and discriminator.
</p>

### 3. Wasserstein GAN (WGAN):
<p>
WGAN is a variant of GAN that introduces the Wasserstein distance as the loss function, leading to more stable training and better convergence properties. It addresses the mode collapse and vanishing gradients issues encountered in traditional GANs.
<br>
WGAN helps to overcome Mode Collapse which occurs when a generative model, fails to capture the diversity of the training data, producing only a limited subset of samples repeatedly, resulting in poor diversity and quality in the generated output.
<br>
Replaces the traditional GAN loss function with the Wasserstein distance. The probability distribution of data and latent noise is compared mostly using Jensen–Shannon divergence but here it is replaced with Wasserstein Distance. In this GAN, the loss has significance while for the above ones, it didn't.
</p>

### 4. Wasserstein GAN with Gradient Penalty (WGAN-GP):
<p>
WGAN-GP is an extension of WGAN that further improves stability by incorporating a gradient penalty term in the loss function. This penalty encourages the discriminator to have Lipschitz continuity, ensuring better training dynamics.
<br>
Replaces the traditional GAN loss function with the Wasserstein distance and adds a gradient penalty term.
</p>

### 5. Conditional GAN (CGAN)
<p>
CGAN is a variant of GAN where both the generator and discriminator are conditioned on additional information such as class labels. This enables control over the characteristics of the generated samples, allowing for targeted generation.
<br>
Adds conditioning information (e.g., class labels) as input to both generator and discriminator. Incorporating labels in both the generator and discriminator helps to produce proper content. Adding a label in the generator will help to produce data of that label and adding the label to the discriminator helps it to identify what type of data to expect and is produced data is of that label as well as good enough for generator to be fooled.   
</p>
