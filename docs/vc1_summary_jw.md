# VideoCrafter1 Summary (Joowhan Song)

## Keywords

Content preservation constraints

Temporal attention layers

Video VAE

Video latent diffusion model

Spatial transformers (ST)

Temporal transformers (TT)

Text-aligned embedding space

----

## Abstract
Two diffusion models for high-quality video generation, T2V, I2V.

This model is the first open-source I2V foundation model capable of transforming a given image into a video clip while maintaining content preservation constraints.

## 1. Introduction
Currently, several open-source T2V models exist, i.e.,ModelScope [50], Hotshot-XL [4], AnimateDiff [23], and Zeroscope V2 XL [7].

The only open-source generic I2V foundation model, I2VGen-XL [15], is released in ModelScope.

However, it does not satisfy the content-preserving constraints. The generated videos match the semantic meaning in the given image but **do not strictly follow the reference content and structure**.

In this work, we introduce two diffusion models for highquality video generation: one for text-to-video (T2V) generation and the other for image-to-video (I2V) generation. The T2V model **builds upon SD 2.1 by incorporating temporal attention layers into the SD UNet to capture temporal consistency.**

Our contributions can be summarized as follows:

• We introduce a text-to-video model capable of generating high-quality videos with a resolution of 1024 × 576 and cinematic quality. The model is trained on 20 million videos and 600 million images.

• We present an image-to-video model, the first opensource generic I2V model that can **strictly preserve the content and structure of the input reference image** while animating it into a video. This model allows for both image and text inputs.

## 2. Related Works

Although T2V models can generate high-quality videos, they only accept text prompts as semantic guidance, which can be verbose and may not accurately reflect users’ intentions.

Although [DragNUWA](https://www.microsoft.com/en-us/research/project/dragnuwa/) [56] further introduce trajectory control into image-to-video generation, which can only mitigate the unrealistic-motion issue to some extent.

![](https://www.microsoft.com/en-us/research/uploads/prod/2023/08/Fig1-64d79c3b13269.gif)

![](https://www.microsoft.com/en-us/research/uploads/prod/2023/08/Fig2-64d8b745bdc9c.gif)

## 3. Methodology
### 3.1. VideoCrafter1: Text-to-Video Model
#### Structure Overview.

The VideoCrafter T2V model is a Latent Video Diffusion Model (LVDM) [24] consisting of two key components: **a video VAE** and **a video latent diffusion model**, as illustrated in Fig. 3.

<img src="https://github.com/YingqingHe/LVDM/blob/main/assets/framework.jpg" alt="lvdm" width="768">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/vc1_fig3.PNG" alt="vc1_fig3" width="768">

The **Video VAE** is **responsible for reducing the sample dimension**, allowing the subsequent diffusion model to be more compact and efficient.

First, the video data `x0` is fed into the VAE encoder `E` to project it into the video latent `z0`, which exhibits a lower data dimension with a compressed video representation. Then, the video latent can be projected back into the reconstructed video `x'0` via the VAE decoder `D`. **We adopt the pretrained VAE from the Stable Diffusion model** to serve as the video VAE and **project each frame individually without extracting temporal information**.

To perform the denoising process, a denoiser U-Net is learned to estimate the noise in the input noisy latent, which will be discussed in the next section.


#### Denoising 3D U-Net.

As illustrated in Fig.3, the denoising U-Net is a 3D U-Net architecture consisting of a stack of basic spatial-temporal blocks with skip connections. Each block comprises convolutional layers, **spatial transformers (ST)**, and **temporal transformers (TT)**, where

$$
ST = \text{Proj}_{in} \circ 
(\text{Attn}\_{self} \circ \text{Attn}\_{cross} \circ \text{MLP}) 
\circ \text{Proj}\_{out}
$$

$$
TT = \text{Proj}\_{in} \circ (\text{Attn}\_{temp} \circ \text{Attn}\_{temp} \circ \text{MLP}) \circ \text{Proj}\_{out}
$$

The controlling signals of the denoiser include semantic control, such as the text prompt, and motion speed control, such as the video fps. We inject the semantic control via the cross-attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V
$$

$$
Q = W_Q^{(i)} \cdot \varphi_i(z_t), \quad K = W_K^{(i)} \cdot \phi(y), \quad V = W_V^{(i)} \cdot \phi(y)
$$

`ϕi` represents spatially flattened tokens of video latent, `φ` denotes the Clip text encoder, and y is the input text prompt.

Motion speed control with fps is incorporated through an FPS embedder, which shares the same structure as the timestep embedder. **Specifically, the FPS or timestep is projected into an embedding vector using sinusoidal embedding.** This vector is then fed into a two-layer MLP to map the sinusoidal embedding to a learned embedding. Subsequently, the timestep embedding and FPS embedding are fused via elementwise addition. 

**The fused embedding is finally added to the convolutional features to modulate the intermediate features.**

### 3.2. VideoCrafter1: Image-to-Video Model

**Text prompts** offer **highly flexible control** for content generation, but they primarily focus on **semantic-level specifications** rather than detailed appearance.

To supply the video model with image information in a compatible manner, **it is essential to project the image into a text-aligned embedding space**. We propose learning such an embedding with rich details to enhance visual fidelity. Figure 4 illustrates the diagram of equipping the diffusion model with an image conditional branch.


<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/vc1_fig4.PNG" alt="vc1_fig4" width="512">

#### Text-Aligned Rich Image Embedding.

Inspired by existing visual conditioning works [44, 55], we utilize the full patch visual tokens Fvis from the last layer of the CLIP image ViT [17], which are believed to encompass much richer information about the image.

To promote alignment with the text embedding, we utilize a learnable projection network P to transform Fvis into the target image embedding Fimg = P(Fvis), enabling the video model backbone to process the image feature efficiently. The text embedding Ftext and image embedding Fimg are then used to compute the U-Net intermediate features
Fin via dual cross-attention layers:

$$
F\_{out} = \text{Softmax}\left(\frac{QK^{T}\_{text}}{\sqrt{d}}\right)V\_{text} + \text{Softmax}\left(\frac{QK^{T}\_{img}}{\sqrt{d}}\right)V\_{img}
$$

Figure 5 compares the visual fidelity of the generated videos conditioned on the global semantic token and our adopted rich visual tokens, respectively.

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/vc1_fig5.PNG" alt="vc1_fig5" width="512">

## 4. Experiments

### 4.1. Implementation Details

#### Datasets. 
We employ **an image and video joint training strategy** for model training. The image dataset used is **LAION COCO** [3], a large text-image dataset consisting of 600 million generated high-quality captions for publicly available web images. For video datasets, we utilize the publicly available **WebVid-10M** [8], a large-scale dataset of short videos with textual descriptions sourced from stock footage sites, offering diverse and rich content. 

Additionally, we compile a large-scale high-quality video dataset containing 10 million videos with resolutions greater than 1280 × 720 for the training of T2V models.

#### Training Scheme.

To train **the T2V model**, we employ **the training strategy used in Stable Diffusion**, i.e., training from low resolution to high resolution. We first train the video model extended from the image model at a resolution of 256 × 256 for 80K iterations with a batch size of 256. Next, we resume from the 256 × 256 model and finetune it with videos for 136K iterations at a resolution of 512×320. The batch size is 128. Finally, we finetune the model for 45K iterations at a resolution of 1024 × 576. The batch size is 64. 

For **the I2V model**, we initially **train the mapping from the image embedding to the embedding space used for the cross attention**. Subsequently, we **fix the mappings of both text and image embeddings** and finetune the video model for improved alignment.

#### Evaluation Metrics.

We employ comprehensive metrics to assess video quality and the alignment between text and video using EvalCrafter [32], a benchmark for evaluating video generation models.

Our T2V model achieves the best visual quality and video quality among open-source models.

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/vc1_fig6.PNG" alt="vc1_fig6" width="256">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/vc1_table1.PNG" alt="vc1_table1" width="256">

### 4.2. Performance Evaluation

Our model encourages large object movements during training, resulting in more significant motion in the generated videos compared to other models.

## 5. Conclusion and Future Work

The existing open-source models merely represent the starting point. Improvements in duration, resolution, and motion quality remain crucial for future developments.

----

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid00.gif" alt="vid00" width="512">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid01.gif" alt="vid01" width="512">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid02.gif" alt="vid02" width="512">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid03.gif" alt="vid03" width="512">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid04.gif" alt="vid04" width="512">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid05.gif" alt="vid05" width="512">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/vids/vid06.gif" alt="vid06" width="512">
