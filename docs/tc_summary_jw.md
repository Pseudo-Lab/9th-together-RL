# ToonCrafter Summary (Joowhan Song)

## Keywords

* generative cartoon interpolation
* toon rectification learning
* live-action motion priors
* dual-reference-based 3D decoder
* frame-independent sketch encoder
* domain gap
* content leakage
* highly compressed latent prior spaces
* hybrid-attention residual learning mechanism
* pseudo-3D convolutions 
* temporal coherence
* adapter mechanism
* [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)

## Key Sentences

In this paper, we point out the importance of generative cartoon interpolation to synthesize frames of complex non-linear motions or phenomena, instead of purely relying on the information from the given frames.

ToonCrafter consists of **three functional techniques**: 

* toon rectification learning, 
* detail injection and propagation in decoding, and 
* sketch-based controllable generation. 

Our contributions are summarized below: 

* We point out the notion of **generative cartoon interpolation** and introduce an innovative solution by leveraging **live-action video prior**.

* We present **a toon rectification learning strategy** that effectively adapts live-action motion prior to animation domain. 

* We propose **a dual-reference-based 3D decoder to compensate the lost details resulting from compressed latent space**. 

* Our system enables users to interactively create or modify interpolation results in a flexible and controllable fashion.

* Even though the cartoon animation might exhibit motions (not as rigid as possible for cases) that are slightly different from the real-world motions, the high-level motion concepts are still the same (otherwise, human viewers cannot recognize what the motion is), making the appearance the dominant factor in domain adaptation for cartoon animation.**

In summary, our toon rectification learning strategy focuses on the appearance by freezing the temporal layers (to preserve the real-world motion prior) and finetuning the image-context projector and spatial layers with only our collected cartoon data to achieve effective domain adaptation.

## Abstract

We introduce ToonCrafter, a novel approach that transcends traditional correspondence-based cartoon video interpolation, paving the way for **generative interpolation**. Traditional methods, that implicitly **assume linear motion** and the absence of **complicated phenomena** like disocclusion, often **struggle with the exaggerated non-linear and large motions with occlusion** commonly found in cartoons, resulting in implausible or even failed interpolation results. 

To overcome these limitations, we explore the potential of adapting **live-action video priors** to better suit cartoon interpolation within a generative framework. ToonCrafter effectively addresses the challenges faced when applying live-action video motion priors to generative cartoon interpolation. 

First, we design **a toon rectification learning strategy** that seamlessly **adapts live-action video priors** to the cartoon domain, **resolving the domain gap and content leakage** issues. 

Next, we introduce **a dual-reference-based 3D decoder** to **compensate for lost details** due to the highly compressed latent prior spaces, ensuring the preservation of fine details in interpolation results. 

Finally, we design **a flexible sketch encoder** that empowers users with interactive control over the interpolation results. Experimental results demonstrate that our proposed method not only produces visually convincing and more natural dynamics, but also effectively handles dis-occlusion.

## 1. Introduction

The differences between cartoon animation and live-action video lie in two aspects, **the frame "sparsity"** and **the texture richness**. While live-action video frames can be densely acquired by camera, cartoon frames are temporally sparse (hence, large motion) due to the high drawing cost. Such cost also leads to higher chance of textureless color regions in cartoon than in live-action video. **Both characteristics make cartoon frame interpolation much challenging.** 

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig01.PNG" alt="tc_fig01" width="512">

Figure 1 shows an example featuring a walking person, apparently linear interpolation can only generate a “drifting” person, instead of a correct walking sequence. Things get even more complicated in a dis-occlusion example in Figure 1(middle). In other words, linear motion assumption is largely insufficient for interpolating obvious motions observed in our daily life. 

**In this paper, we point out the importance of generative cartoon interpolation to synthesize frames of complex non-linear motions or phenomena, instead of purely relying on the information from the given frames.**

Unfortunately, directly applying existing models to cartoon interpolation is unsatisfactory due to **three factors**. 

* Firstly, there exists **a domain gap** as the models are mostly trained on live-action video content. 

* Secondly, to reduce the computational cost, current video diffusion models are based on highly compressed latent spaces, resulting in significant loss of details and quality degradation. 

* Lastly, the generative models can be somewhat random and lack of control. An effective control over the generated motion is necessary for cartoon interpolation. 

In this paper, we propose an effective and controllable generative framework, ToonCrafter, to adapt the pre-trained video diffusion model, and address the three challenges above. 

ToonCrafter consists of **three functional techniques**: 

* toon rectification learning, 
* detail injection and propagation in decoding, and 
* sketch-based controllable generation. 

Specifically, the toon rectification learning strategy involves the meticulous fine-tuning of the **spatial-related context understanding** and **content generation layers** of the underlying image-conditioned video generation model on collected cartoon data.

**To tackle the detail loss and quality degradation issue, we introduce a dual reference-based 3D decoder**, featuring **a hybrid-attention residual learning mechanism**, to convert lossy frame latents back to pixel space. It injects the detail information from input images into the generated frame latents using **a cross-attention mechanism in shallow decoding layers** and **residual learning in deeper layers**, considering computational cost burden. Furthermore, the decoder is equipped with **pseudo-3D convolutions to facilitate propagation and improve the temporal coherence**. 

Lastly, we propose **a frame independent sketch encoder** that enables users to flexibly and interactively create or modify interpolation results with **temporally sparse or dense motion structure guidance**. It also allows users to effectively **control the generated motion via sparse sketch input.** 

Our contributions are summarized below: 

* We point out the notion of **generative cartoon interpolation** and introduce an innovative solution by leveraging **live-action video prior**.

* We present **a toon rectification learning strategy** that effectively adapts live-action motion prior to animation domain. 

* We propose **a dual-reference-based 3D decoder to compensate the lost details resulting from compressed latent space**. 

* Our system enables users to interactively create or modify interpolation results in a flexible and controllable fashion.

## 2. Related Work

### 2.1. Video Frame Interpolation

Video frame interpolation aims at synthesizing multiple frames in between two adjacent frames of the original video, which has been widely studied in recent years. 

Existing works using deep learning fall into three categories, including 
* phase-based, 
* kernel-based, and 
* optical/feature flow-based methods. 

The most recent state-of-the-art has seen more optical flow-based methods, benefited from the latest advancements in flow estimation. **The typical approach first identifies the correspondence between two frames using flow, and then performs warping and fusion.** Readers are recommended to refer to [11](https://dl.acm.org/doi/10.1145/3556544) for a comprehensive survey.

### 2.2. Image-conditioned Video Diffusion Models

In this paper, we aim to leverage the rich motion-generative prior in I2V diffusion models learned from live-action videos and adapt it for generative cartoon interpolation.

## 3. Method

**Our generative cartoon interpolation framework is built upon the open-sourced DynamiCrafter interpolation model**, a state-of-the-art image-to-video generative diffusion model that demonstrates robust motion understanding for live-action interpolation but falls short when applied to cartoon animations. 

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig02.PNG" alt="tc_fig02" width="384">

As shown in Figure 2, our framework shares a similar structure to the base model but incorporates three key improvements for generative cartoon interpolation: 
* (1) a meticulously designed toon rectification learning strategy for effective domain adaptation, 
* (2) **a novel dual-reference 3D decoder D** to tackle the visual degradation due to the lossy latent space, and 
* (3) **a frame independent sketch encoder S** that enables the user control over the interpolation.

### 3.1. Preliminary

Given a video x ∈ R^{L×3×H×W}, we first encode it into a latent representation z = E(x), z ∈ R^{L×C×h×w} on a frame-by-frame basis. **Next, both the forward diffusion process z_t = p(z_0, t) and backward denoising process z_t = p_θ(z_{t−1}, c, t) are executed in this latent space, where c represents the denoising conditions such as text c_{txt} and image prompts c_{img}**. Following the description in DynamiCrafter, the interpolation application is realized by providing both starting and ending frames x_1 and x_L while leaving middle frames as empty for c_{img}. 

Then, the objective is:

$$
\min_{\theta} \mathbb{E}_{E(x), t, \epsilon \sim \mathcal{N}(0, I)} \left[ \|\epsilon - \epsilon_{\theta} (z_t; c_{\text{img}}, c_{\text{txt}}, t, \text{fps}) \|_2^2 \right]
$$

where fps is the FPS control introduced in [54]. The generated videos are then obtained through the decoder ˆx = D(z_0).

### 3.2. Toon Rectification Learning

To address this, we first collect a cartoon video dataset and then **adapt the motion prior model to the cartoon domain** by meticulously designed fine-tuning.

**Cartoon Video Dataset Construction.**

We collect a series of raw cartoon videos and then manually select high-quality ones based on the resolution and subjective quality. The total duration of the selected videos is more than 500 hours. We employ PySceneDetect [1] to detect and split shots. The static shots are filtered out by removing any videos with low average optical flow [46] magnitude. Moreover, we apply optical character recognition (CRAFT) [2] to weed out clips containing large amounts of text. In addition, we adopt LAION [42] regressive model to calculate the aesthetic score for removing the low-aesthetic samples to ensure quality. Next, we annotate each clip with the synthetic captioning method BLIP-2 [23]. Lastly, we annotate the first, middle, and last frames of each video clip with CLIP [39] embeddings from which we measure the text-video alignment, to filter out mismatched samples. In the end, we obtained 271K high-quality cartoon video clips, which were randomly split into two sets. The training set contains 270K clips, while the evaluation set contains 1K clips.

**Rectification Learning.**

With the collected cartoon text-video data, we can then adapt the DynamiCrafter interpolation model (DCinterp) trained on live-action videos for cartoon interpolation. However, directly fine-tuning the denoising network of DCinterp on our data would lead to catastrophic forgetting due to unbalanced scale between our cartoon video data (270K video clips) and the original training data of DCinterp (WebVid-10M [3], 10M), which deteriorates motion prior, as evidenced in **Sec. 4.5**. 

To address this issue, we design an efficient rectification learning strategy that allows for fine-tuning the base model using only a small-scale cartoon dataset without compromising the robust motion prior obtained from the large-scale live-action videos. 

The DCinterp model consists of **three key components**: 
* an image-context projector, 
* the spatial layers (sharing the same architecture as StableDiffusion v2.1), and 
* the temporal layers. 

Based on our experiments (Sec. 4.5), we have the following observations: 
* the image-context projector helps the DCinterp model to **digest the context** of the input frames; 
* the spatial layers are responsible for learning the appearance distribution of video frames; 
* the temporal layers capture the motion dynamics between the video frames. 

In other words, **the temporal layers should be frozen** during the fine-tuning to preserve the real-world motion prior as illustrated in Figure 2. On the other hand, **the image-context projector can be fine-tuned** to achieve better semantic alignment and allow the model to digest the cartoon scene context better. Simultaneously, **spatial layers should also be tuned for appearance rectification**, thereby preventing the generation of live-action video content in the intermediate frames. 

**Even though the cartoon animation might exhibit motions (not as rigid as possible for cases) that are slightly different from the real-world motions, the high-level motion concepts are still the same (otherwise, human viewers cannot recognize what the motion is), making the appearance the dominant factor in domain adaptation for cartoon animation.**

**In summary, our toon rectification learning strategy focuses on the appearance by freezing the temporal layers (to preserve the real-world motion prior) and finetuning the image-context projector and spatial layers with only our collected cartoon data to achieve effective domain adaptation.**

### 3.3. Detail Injection and Propagation in Decoding

Most of the current video diffusion models, including DynamiCrafter we built upon, learn to generate the videos in highly compressed latent spaces, which are typically obtained through **vector quantized auto-encoding (VQ-VAE)**. The latent video diffusion paradigm effectively reduces computational demand. However, it results in a significant loss of details and intolerable quality degradation including flickering and distortion artifacts. Unfortunately, such degradations are more apparent in cartoon animation due to its typical appearance of high-contrast regions, fine structural outline, and the lack of motion blur (motion blur in live-action video effectively “hides” the degradation).

To address this issue, we propose to **exploit the existing information from the input frames** and introduce **a dual reference-based 3D decoder** to **propagate the pixel-level details** from the two input frames to the decoding process of the generated lossy-space frame latents. 

Rather than **relying solely on the decoder D** to recover the compressed details, **we firstly extract the inner features {F^K_i} at each residual block of E** (where i represents **the i-th residual block from the end in the encoder** and K indicates the K-th frame), and then **inject them into the decoding process**. This provides the necessary hints for achieving pixel-perfect compensation. 

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig03.PNG" alt="tc_fig03" width="384">

Specifically, we propose **a hybrid-attention-residual-learning mechanism (HAR)** to inject and propagate details. As shown in Figure 3, **we introduce cross-frame-attention in D** to inject the intricate details from {F^1_i}i∈s and {F^L_i}i∈s to decoder’s intermediate features G_in: 

$$
G^{\text{out}}_j = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V + G^{\text{in}}_j, \quad j \in \{1, \dots, L\}
$$

where Q = [G^j_in]W_Q, K = [F^1_i;F^L_i]W_K, V = [F^1_i;F^L_i]W_V, and [;] denotes concatenation. Considering the computational cost of attention, we implement it only in the first two layers (i.e., shallow layers s = {1, 2}) of D. Since the input frame x1 and resultant ˆx1 are aligned in pixel-level, we add the ZeroConv [58] processed {F1i }i∈d (ZeroConv processed {FLi }i∈d) to the corresponding feature maps of the first frame (the L-th frame)

$$
G^1_{\text{out}} = \text{ZeroConv}_{1 \times 1}(F^1_i) + G^1_{\text{in}}
$$

To avoid redundant computation, we implement this **residual learning only at the deep layers (d = {3, 4, 5}) of D**. In addition, we incorporate **pseudo-3D convolutions (P3D)** to further facilitate the propagation and improve the temporal coherence. 

**Training.**

**We freeze the image encoder E and optimize the proposed decoder D, which is initialized from the vanilla image decoder.** We use a compound loss L to encourage reconstruction: 

$$
L = L_1 + \lambda_p L_p + \lambda_d L_d
$$

where L1 is the MAE loss, Lp is a perceptual loss (LPIPS [59]), Ld is a discriminator loss [26], and λp = 0.1, λd is an adaptive weight following [12].

### 3.4. Sketch-based Controllable Generation

To make our framework more controllable for real-world production settings, we follow the industrial practices [25] and introduce sketch-based generation guidance. **We propose a frame-independent sketch encoder S that enables users to control the generated motion using sparse sketch guidance.** Built upon **the adapter mechanism [58]**, our sketch encoder effectively converts our video diffusion model into a sketch-conditioned generative model.

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig04.PNG" alt="tc_fig04" width="384">

To reduce users’ drawing efforts and accommodate realworld usage, we design our sketch encoder module S that supports sparse inputs (Figure 4) in which the users are not required to provide all sketch images for the target frames. To achieve this, we **design S as a frame-wise adapter that learns to adjust the intermediate features of each frame independently based on the provided sketch**: Fi inject = S(si, zi, t), where **Fi inject is processed using the same strategy of ControlNet [58]**. For the frames without sketch guidance, S takes an empty image as input: Fi inject = S(s∅, zi, t). Using empty image inputs improves the learning dynamics of sketch encoder.

**Training.**
We freeze the denoising network ϵθ and optimize the sketch encoder S. S uses a ControlNet-like architecture, initialized from the pre-trained StableDiffusion v2.1. The training objective is:

$$
\min_{\theta} \mathbb{E}_{(x), s, t, \epsilon \sim \mathcal{N}(0, I)} \| \epsilon - \epsilon^S_{{\theta}} (z_t; c_{\text{img}}, c_{\text{txt}}, s', t, \text{fps}) \|_2^2
$$

where ϵSθ denotes the combination of ϵθ and S, s denotes sketches obtained from **Anime2Sketch [51]** using original video frames, and s′ denotes selected sketches from s (illustrated in Figure 4). To support typical patterns of user provided sketch inputs, we design a bisection selection pattern (chosen 80% of the time) to select input sketches: for an interpolation segment (i, j), the sketch of ⌊ (i+j) / 2 ⌋-th frame is selected; the selection is applied recursively (with the recursion depth n uniformly sampled from [1, 4]) from segment (1,L) to subdivided segments. This bisection selection pattern mimics real-world user behavior, where users provides sketches at equally-spaced interval. For the remaining 20%, we randomly select input sketches from s to maximize the generalization ability.

## 4. Experiments

### 4.1. Implementation Details

Our implementation is primarily based on the image-to-video model DynamiCrafter [54] (interpolation variant @512×320 resolution).

### 4.2. Quantitative Comparisons

### 4.3. Qualitative Comparisons

### 4.4. User Study

The participants were asked to choose the best result in terms of **motion quality, temporal coherence, and frame fidelity**.

### 4.5. Ablation Study

**Toon rectification learning.** 

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig05.PNG" alt="tc_fig05" width="384">

We construct the following baselines to investigate the effectiveness of our domain adaptation strategy: 

(I): directly using the pre-trained backbone model (DCinterp [54]), 

(II): fine-tuning the imagecontext projector (ICP) and entire denoising U-Net (Spatial+ Temporal layers), 

(III): fine-tuning ICP and spatial layers while bypassing temporal layers in forwarding during training, 

(IV) (our adapting strategy): fine-tuning ICP and spatial layers while keeping temporal layers frozen , and 

(V): fine-tuning only ICP. 

The quantitative comparison is shown in Table 3. DCInterp without any fine-tuning (I) shows decent quantitative results but suffers from **unexpected generation of live-action content** (2nd row in Figure 5). While directly fine-tuning all layers (II) leads to adaption to cartoon domain to some extent, **it deteriorates the temporal prior, as evidenced by the inconsistency and sudden change of generated content** (3rd row in Figure 5). Moreover, simply bypassing temporal layers (III) in forwarding to preserve temporal prior leads to disastrous degradation due to mismatched distribution mapping. Comparing (I), (II), and (IV), we can observe improved performance of both FVD and CLIPimg by fine-tuning ICP and spatial layers, while keeping temporal layers frozen, which enhances the adaptability to cartoon domain and preserves learned motion prior. The comparison between (I) and (V) shows fine-tuning ICP slightly improves semantic alignment for generating semantically correct content (higher CLIPimg), thanks to its better comprehension of cartoon context. 

**Dual-reference-based 3D VAE decoder.**

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig06.PNG" alt="tc_fig06" width="384">

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig08.PNG" alt="tc_fig08" width="384">

We further evaluate the effectiveness of different modules in the proposed dual-reference-based 3D decoder. We first construct a variant by **removing the pseudo-3D convolutions (P3D)**, denoted by Oursw/o P3D. Base on that, we then further remove the introduced **hybrid-attention-residual (HAR) module** to obtain the second variant Oursw/o HAR & P3D, which is exactly the image decoder used in most diffusion-based image/video generative models. We evaluate the our full method with the mentioned variant via video reconstruction task and report the evaluation results in Table 4. The performance of ‘Oursw/o P3D’ declines due to the attenuation of propagation for injected reference information, leading to a loss of details (3rd column in Figure 6). Furthermore, removing both HAR and P3D considerably impairs the performance, as shown in Table 4, since solely relying on the VAE decoder without reference fails to recover lost details in compressed latents (4th column in Figure 6). In contrast, our full method effectively compensates for the lost details through the introduced dual-reference-based detail injection and propagation mechanism. Additionally, we show the comparison of reconstruction quality along frame sequences (average on the 1K test samples) in Figure 8, which further highlights the effectiveness of our design. 

**Sparse sketch guidance.** 

<img src="https://github.com/Pseudo-Lab/9th-together-RL/blob/main/imgs/tc_fig07.PNG" alt="tc_fig07" width="384">

To verify the design of our frame-independent sketch encoder, we construct a variant by training S with full conditions s (i.e., per-frame sketch guidance) and enable its sparse control by zeroing out Fi inject for frames without guidance during inference. We provide only the middle-frame sketch as sparse control and compare this **ZeroGate** variant with our **FrameIn.Enc.** (frame independent encoder), as shown in Figure 7. Although ‘ZeroGate’ can generate the middle frame adhering to the sketch guidance, it struggles to produce consistent content for other unguided frames. In contrast, our ‘FrameIn.Enc.’ not only generates middle frames with good conformity to the sketch, but also maintains temporal coherence across the generated sequence. We also present the generated results without sketch guidance (4th row in Figure 7) using the same input cartoon images.

**Toon rectification learning.**

**Dual-reference-based 3D VAE decoder.**

**Sparse sketch guidance.**

## 5. Applications

**Cartoon sketch interpolation** is more than challenging due to its extremely sparse structure without color and textures. **Nonetheless, our ToonCrafter can still produce decent results (Figure 10 top) on such unseen input, thanks to its powerful generalization capabilities.** It can also **support reference-based sketch colorization by providing 1 or 2 reference images** and per-frame sketches. The visual results of these applications are presented in Figure 10.

## 6. Conclusion

We introduced ToonCrafter, an innovative framework for the first attempt of **generative cartoon interpolation**. We propose **the toon rectification learning** to retain **the live-action motion priors** while overcoming the domain gap, and preserve the visual details through **the dual-reference-based 3D decoder**. To allow user control over the interpolation, we design **a frame-independent sketch encoder**.