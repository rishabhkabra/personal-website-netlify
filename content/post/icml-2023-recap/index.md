---
title: ICML 2023 Recap
subtitle: My top 50 posters from ICML Honolulu.

# Summary for listings and search engines
summary: A recap from ICML 2023 comprising 50 posters organized by topic, e.g., vision, generative models.

# Link this post with a project
projects: []

# Date published
date: '2023-07-31T00:00:00Z'

# Date updated
lastmod: '2023-07-31T00:00:00Z'

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  focal_point: ''
  placement: 2
  preview_only: false

authors:
  - admin

categories:
  - Conferences
---

## Vision

### Vision Transformers

- VIT-22B models have much better alignment with human visual perception: 87% shape bias versus 20-30% in prior models. Prior models were much more texture-biased.
![alt_text](posters/dehghani.jpg)

- A hierarchical VIT i.e. non-uniform feature size through the depth of the network. Also removes unnecessary bells and whistles from prior work by learning those biases instead.
![alt_text](posters/ryali.jpg)

- VIT with global attention interspersed with regular attention
![alt_text](posters/hatamizadeh.jpg)

### 2D

- Use both text and vision to improve classification of novel classes.
![alt_text](posters/kaul.jpg)

- Learning a displacement field to learn the correspondence between photos and sketches
![alt_text](posters/lu_x.jpg)

- Interpretable subspaces in image representations extracted using CLIP
![alt_text](posters/kalibhat.jpg)

- Measuring *compositionality* and *invertibility* for object-centric representations
![alt_text](posters/brady.jpg)

- Multi-view self-supervised learning analyzed using Mutual Information.
![alt_text](posters/galvez.jpg)

- Class collapse and feature suppression during contrastive learning
![alt_text](posters/xue_y.jpg)

- The latest on hyperbolic representations.
![alt_text](posters/desai.jpg)

### 3D

- Spherical CNNs (rotation equivariant) scaled to 5e6 convolutions and 1e7-1e9 feature maps
![alt_text](posters/esteves.jpg)

- Object pose canonicalization measured for *stability* and *consistency*. They also train on multiple object classes.
![alt_text](posters/kim_s.jpg)

- Signed distance functions learnt “provably.”
![alt_text](posters/bethune.jpg)

### Video

- Keypoint learning in videos.
![alt_text](posters/younes.jpg)

- Efficient episodic recall (aka “video search”).
![alt_text](posters/ramakrishnan.jpg)


## Generative Models

- Electrostatics-based generative model with better FID numbers than diffusion
![alt_text](posters/xu_y.jpg)

- Animated 3D models without any additional dataset.
![alt_text](posters/singer.jpg)

- Diffusion without upsamplers. Harder to train and inefficient.
![alt_text](posters/hoogeboom.jpg)

- Consistency models: diffusion without multi-step denoising.
![alt_text](posters/song_y.jpg)

- Diffusion models evaluated on one-shot drawing task.
![alt_text](posters/boutin.jpg)

- NeRF from fewer samples using geometric invariances.
![alt_text](posters/kwak.jpg)

## World Models/RL
![alt_text](posters/wu_p.jpg)

![alt_text](posters/freed.jpg)

![alt_text](posters/seo_y.jpg)

![alt_text](posters/nottingham.jpg)

![alt_text](posters/gmelin.jpg)

![alt_text](posters/ghosh.jpg)

## Transformers

- Beautiful work showing transformers have a “lower-degree” bias toward polynomial terms of lower degree, which is somewhat counterintuitive given their pairwise attention mechanism. 
![alt_text](posters/abbe.jpg)

- Improving the focal loss by taking into account the second highest predicted logit, rather than naively maximizing entropy.
![alt_text](posters/tao_l.jpg)

- Do early layers generalize while later layers memorize? Apparently not–memorization can be localized to a small number of neurons dispersed across layers.
![alt_text](posters/maini.jpg)

- Characterizing training trajectories of different representation learning tasks
![alt_text](posters/ramesh.jpg)

- Is local flatness desirable for generalization? Not necessarily. There are more promising indicators such as SGD-based disagreement on unlabelled data.
![alt_text](posters/andriushchenko.jpg)

- Category-theory view of disentanglement
![alt_text](posters/zhang_y.jpg)

- Using category theory to show that foundation models cannot be used for everything, but CLIP-like algorithms do have “creativity”
![alt_text](posters/yuan_y.jpg)

## Novel architectures

- Super simple long convolutions
![alt_text](posters/fu_d.jpg)

- Differentiable “if blocks”
![alt_text](posters/faber.jpg)

- Differentiable tree operations
![alt_text](posters/soulos.jpg)

- Continuous spatiotemporal transformers
![alt_text](posters/fonseca.jpg)

## Graphs

- Compositionality via learnt pooling from a multi-view graph to a latent graph
![alt_text](posters/liu_t.jpg)

- Positional encodings to take advantage of edge directions
![alt_text](posters/geisler.jpg)

![alt_text](posters/ma_l.jpg)

## Adversarial attacks

- Independent component analysis to design an attack on federated learning
![alt_text](posters/kariyappa.jpg)

![alt_text](posters/khaddaj.jpg)

## Curiosities

![alt_text](posters/becker.jpg)

- Implicit neural representations (using spatial coordinates C or environmental features E or both) to predict presence of wildlife species.
![alt_text](posters/cole.jpg)

![alt_text](posters/huang_l.jpg)

- ML on Mars for source separation to detect marsquakes!
![alt_text](posters/siahkoohi.jpg)

- Template + score/filter prompts for a dataset without access to labels.
![alt_text](posters/allingham.jpg)

- A simple initialization trick for VIT-Tiny
![alt_text](posters/trockman.jpg)

- How to fine-tune ML models in an “open-source” fashion: fine-tune in parallel and then merge
![alt_text](posters/rame.jpg)
