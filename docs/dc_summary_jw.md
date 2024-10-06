# DynamiCrafter Summary (Joowhan Song)

## Keywords

the motion prior

a query transformer

text-aligned rich context representation space

context learning network

dual-stream injection paradigm

## Abstract

Traditional image animation techniques mainly focus on animating natural scenes with tochastic dynamics (e.g. clouds and fluid) or domain-specific motions (e.g. human hair or body motions), and thus **limits their applicability to more general visual content**. 

To overcome this limitation, **we explore the synthesis of dynamic content for open-domain images**, converting them into animated videos. 

The key idea is to utilize **the motion prior** of text-to-video diffusion models by incorporating the image into the generative process as **guidance**. 

Given an image, we first project it into a text-aligned rich context representation space using **a query transformer**, which facilitates the video model to digest the image content in a compatible fashion. 

To supplement with more precise image information, we further feed the full image to the diffusion model by concatenating it with the initial noises. 

## 1. Introduction

Our key idea is to govern the video generation process of T2V diffusion models by incorporating a conditional image.

To address this challenge, we propose **a dual-stream image injection paradigm**, comprised of **text-aligned context representation** and **visual detail guidance**, which ensures that the video diffusion model synthesizes detail-preserved dynamic content in a complementary manner.

Given an image, we first project it into the **text-aligned rich context representation space** through a specially designed **context learning network**.

Specifically, it consists of **a pre-trained CLIP image encoder** to extract **text-aligned image features** and **a learnable query transformer** to further promote its adaptation to the diffusion models.

The rich context features are used by the model via **cross attention layers**, which will then be combined with the text-conditioned features through **gated fusion**. In some extend, **the learned context representation trades visual details with text alignment** which helps facilitate semantic understanding of image context so that reasonable and vivid dynamics could be synthesized. To supplement more precise visual details, we further feed the full image to the diffusion model by concatenating it with the initial noise. This **dual-stream injection paradigm** guarantees both **plausible dynamic content and visual conformity** to the input image.

Furthermore, we offer discussion and analysis on some insightful designs for diffusion model based image animation, such as the roles of different visual injection streams, the utility of text prompts and their potential for dynamics control, which may inspire follow-ups to push forward this line of technique.

Our contributions are summarized as follows:
• We introduce an innovative approach for animating open-domain images by leveraging video diffusion prior, significantly outperforming contemporary competitors.
• We conduct a comprehensive analysis on the conditional space of text-to-video diffusion models and propose a dual-stream image injection paradigm to achieve the challenging goal of image animation.
• We pioneer the study of text-based motion control for open-domain image animation and demonstrate the proof of concept through preliminary experiments.

## 2. Related Work

### 2.1. Image Animation
In contrast, our work proposes a generic framework for animating open-domain images with a wide range of content and styles, which is extremely challenging due to the overwhelming complexity and vast diversity.

### 2.2. Video Diffusion Models
To replicate this success to video generation, the first video diffusion model (VDM) [30] is proposed to model low-resolution videos using a spacetime factorized U-Net in pixel space.

Our approach is built upon **text-conditioned VDMs** to leverage their rich dynamic prior for animating open-domain images, by incorporating tailored designs for better semantic understanding and conformity to the input image.

## 3. Method
Given a still image, we aim at animating it to produce a short video clip, that inherits all the visual content from the image and exhibits an implicitly suggested and natural dynamics.

We tackle this synthesis task by utilizing the generative priors of pre-trained video diffusion models.

### 3.1. Preliminary: Video Diffusion Models

In this paper, our study is conducted based on an open-source video LDM VideoCrafter [8].

Given a video *x* ∈ ℝ^L×3×H×W, we first encode it into a latent representation *z* = E(*x*), *z* ∈ ℝ^L×C×h×w frame-by-frame. Then, both the forward diffusion process *zₜ* = *p(z₀, t)* and backward denoising process *zₜ* = *pₜₕₑₜₐ(*zₜ₋₁*, *c*, *t*)* are performed in this latent space, where *c* denotes possible denoising conditions like a text prompt. Accordingly, the generated videos are obtained through the decoder *x̂* = D(*z*).

### 3.2. Image Dynamics from Video Diffusion Priors

On the one hand, the image should be digested by the T2V model for context understanding, which is important for dynamics synthesis. On the other, the visual details should be preserved in the generated videos. Based on this insight, we propose a dual-stream conditional image injection paradigm, consisting of text-aligned context representation and visual detail guidance. The overview diagram is illustrated in Figure 1.

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/dc_fig1.PNG" alt="dc_fig1" width="768">

### Text-aligned context representation.
Since the text embedding is constructed with pre-trained CLIP [56] text encoder, we employ the image encoder counterpart to extract image feature from the input image.

Although **the global semantic token f^cls from the CLIP image encoder** is well-aligned with image captions, it mainly represents the visual content at the semantic level and **fails to capture the image’s full extent**.

To extract more complete information, we use the full visual tokens from the last layer of the CLIP image ViT [15], which demonstrated high-fidelity in conditional image generation works [61, 88].

To promote the alignment with text embedding, in other words, to obtain a context representation that can be interpreted by the denoising U-Net, **we utilize a learnable lightweight model P to translate Fvis into the final context representation Fctx = P(Fvis)**. We employ the query transformer architecture [1, 35] in multimodal fusion studies as P, which comprises **N stacked layers of cross-attention and feed-forward networks (FFN), and is adept at cross-modal representation learning via the cross-attention mechanism**.

Subsequently, the text embedding Ftxt and context embedding Fctx are employed to interact with the U-Net intermediate features Fin through the dual cross-attention layers:

$$
F_{out} = \text{Softmax}\left(\frac{QK^{\top}}{\sqrt{d_{txt}}} \right) V_{txt} + \lambda \cdot \text{Softmax}\left( \frac{QK^{\top}}{\sqrt{d_{ctx}}} \right) V_{ctx}
$$

In particular, λ denotes the coefficient that fuses text-conditioned and image-conditioned features, which is achieved through tanh gating and adaptively learnable for each layers. This design aims to facilitate the model’s aility to absorb image conditions in a layer-dependent manner.

### Observations and analysis of λ.

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/dc_fig2.PNG" alt="dc_fig2" width="512">

Figure 2 (left) illustrates the learned coefficients across different layers, indicating
that the image information has a more significant impact on the two-end layers w.r.t. the intermediate layers. To explore further, we manually alter λ in the intermediate layers. As depicted in Figure 2 (right), **increasing λ leads to suppressed cross-frame movements, while decreasing λ poses challenges in preserving the object’s shape**.

### Visual detail guidance (VDG).

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/dc_fig3.PNG" alt="dc_fig3" width="512">

The rich-informative context representation enables the video diffusion model to produce videos that closely resemble the input image. However, as shown in Figure 3, **minor discrepancies** may still occur. This is mainly due to **the pre-trained CLIP image encoder’s limited capability to fully preserve input image information**, as it is designed to align **visual and language features**.

To enhance visual conformity, we propose providing the video model with additional visual details from the image. Specifically, we **concatenate the conditional image with per-frame initial noise and feed them to the denoising U-Net as a form of guidance**.

### Discussion.
(i) Why are text prompts necessary when a more informative context representation is provided?
Additional text prompts can offer a native global context that enables the model to efficiently utilize image information. Figure 3 (right) demonstrates how incorporating text can address the issue of shape distortion in the bear’s head. Furthermore, as a still image typically contains multiple potential dynamic variations, text prompts can effectively guide the generation of dynamic content tailored to user preferences (see Sec. 5).

(ii) Why is a rich context representation necessary when the visual guidance provides the complete image?
The corresponding ablation study is presented in Sec. 4.4.

### 3.3. Training Paradigm

To modulate them in a cooperative manner, we device a dedicated training strategy consisting of three stages, i.e., 

**(i) training the image context representation network P,**

**(ii) adapting P to the T2V model, and**

**(iii) joint fine-tuning with VDG.**

Specifically, to offer the image information to the T2V model in a compatible fashion, we propose to train a context representation network P to extract text-aligned visual
information from the input image. 

Considering the fact that P takes numerous optimization steps to converge, **we propose to train it based on a lightweight T2I model instead of a T2V model**, allowing it to focus on image context learning, and then **adapt it to the T2V model by jointly training P and spatial layers (in contrast to temporal layers) of the T2V model**. 

After establishing a compatible context conditioning branch for T2V, we concatenate the input image with per-frame noise for joint fine-tuning to enhance visual conformity. Here we only **fine-tune P and the VDM’s spatial layers to avoid disrupting the pre-trained T2V model’s temporal prior knowledge with dense image concatenation**, *which could lead to significant performance degradation and contradict our original intention.*

Additionally, we **randomly select a video frame** as the image condition based on two considerations: 

(i) to prevent the network from learning a shortcut that maps the concatenated image to a frame in the specific location, and 

(ii) to force the context representation to be more flexible to avoid offering the over-rigid information for a specific frame, i.e., the objective in the context learning based on T2I.

## 4. Experiment

### 4.1. Implementation Details

Our development is based on the open-source T2V model VideoCrafter [8] (@256 × 256 resolution) and T2I model Stable-Diffusion-v2.1 (SD) [58].

### 4.2. Quantitative Evaluation

### Metrics and datasets. 
To evaluate the quality and temporal coherence of synthesized videos in both the spatial and temporal domains, we report Frechet Video Distance (FVD) [72] as well as Kernel Video Distance (KVD) [72]. 

To further investigate the perceptual conformity between the input image and the animation results, we introduce Perceptual Input Conformity (PIC).

### 4.3. Qualitative Evaluation

Figure 4 presents the visual comparison of image animation results with various content and styles. Among all compared methods, our approach generates temporally coherent videos that adhere to the input image condition. It is worth noting that our method allows dynamic control through text prompts while other methods suffers from neglecting the text modality (e.g., talking in the ‘Girl’ case).

### 4.4. Ablation Studies

### Dual-stream image injection.

To investigate the roles of each image conditioning stream, we examine two variants:

i). Ours w/o ctx, by removing the context conditioning stream, 

ii). Ours w/o VDG, by removing the visual detail guidance stream.

Table 3 presents a quantitative comparison between our full method and these variants.

### Training paradigm.

We firstly construct a baseline by training the context representation network P based on the pre-trained T2V and keeping other settings unchanged.

## 5. Discussions on Motion Control using Text

Since images are typically associated with multiple potential dynamics in its context, text can complementarily guide the generation of dynamic content tailored to user preference. However, captions in existing large-scale datasets often consist of a combination of a large number of scene descriptive words and less dynamic/motion descriptions, potentially causing the model to overlook dynamics/motions during learning. For image animation, the scene description is already included in the image condition, while the motion description should be treated as text condition to train the model in a decoupled manner, providing the model with stronger text-based control over dynamics.

### Dataset construction.

## 6. Applications

**i). Storytelling with shots.** 

First, we utilize ChatGPT (equipped with DALL-E 3 [62]) to generate a story script and corresponding shots (images). And then storytelling videos can be generated by animating those shots with story scripts using DynamiCrafter, as displayed in Figure 10 (top). 

**ii). Looping video generation.** 

With minor modifications, our framework can be adapted to facilitate the generation of looping videos. Specifically, we provide both x1 and xL as visual detail guidance and leave other frames as empty during training. During inference, we set both of them as the input image. Additionally, we experiment with building this application on top of a higher resolution (320×512) version of VideoCrafter. The looping video result is shown in Figure 10 (middle). 

**iii). Generative frame interpolation.**

Furthermore, the modified model enables generative frame interpolation by set the input images x1 and xL differently, as shown in Figure 10 (bottom).

## 7. Conclusion

In this study, we introduced DynamiCrafter, an effective framework for animating open-domain images by leveraging pre-trained video diffusion priors with the proposed dual-stream image injection mechanism and dedicated training paradigm. 

Our experimental results highlight the effectiveness and superiority of our approach compared to existing methods. 

Furthermore, we explored textbased dynamic control for image animation with the constructed dataset. 

Lastly, we demonstrated the versatility of our framework across various applications and scenarios.

## Supplementary Material