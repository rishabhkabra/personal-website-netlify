---
title: 'SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Daniel Zoran
  - Goker Erdogan
  - Loic Matthey
  - Antonia Creswell
  - Matt Botvinick
  - Alexander Lerchner
  - Chris Burgess

date: '2021-12-06T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: In Advances in Neural Information Processing Systems
publication_short: In *NeurIPS*

abstract: "To help agents reason about scenes in terms of their building blocks, we wish to extract the compositional structure of any given scene (in particular, the configuration and characteristics of objects comprising the scene). This problem is especially difficult when scene structure needs to be inferred while also estimating the agent’s location/viewpoint, as the two variables jointly give rise to the agent’s observations. We present an unsupervised variational approach to this problem. Leveraging the shared structure that exists across different scenes, our model learns to infer two sets of latent representations from RGB video input alone: a set of 'object' latents, corresponding to the time-invariant, object-level contents of the scene, as well as a set of 'frame' latents, corresponding to global time-varying elements such as viewpoint. This factorization of latents allows our model, SIMONe, to represent object attributes in an allocentric manner which does not depend on viewpoint. Moreover, it allows us to disentangle object dynamics and summarize their trajectories as time-abstracted, view-invariant, per-object properties. We demonstrate these capabilities, as well as the model's performance in terms of view synthesis and instance segmentation, across three procedurally generated video datasets."

# Summary. An optional shortened abstract.
summary: A video scene model which separates the time-invariant, object-level contents of the scene from global time-varying elements such as viewpoint.

tags: []

# Display this page in the Featured widget?
featured: true

url_slides: 'https://dpmd.ai/simone-slides'
url_project: 'https://sites.google.com/view/simone-scene-understanding/'
url_video: 'https://slideslive.com/38968266/simone-viewinvariant-temporallyabstracted-object-representations-via-unsupervised-video-decomposition'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.

---

