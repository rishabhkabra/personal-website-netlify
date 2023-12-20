---
title: NeurIPS 2023 Recap
subtitle: A collection of posters which caught my eye, reflecting 2% of NeurIPS.

# Summary for listings and search engines
summary: A recap from NeurIPS 2023 covering topics like VLMs, scene understanding, object-centric representations, and generative models.

# Link this post with a project
projects: []

# Date published
date: '2023-12-18T00:00:00Z'

# Date updated
lastmod: '2023-12-18T00:00:00Z'

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
  - Conference
---

<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 85.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>
<a href="#gdcalert4">alert4</a>
<a href="#gdcalert5">alert5</a>
<a href="#gdcalert6">alert6</a>
<a href="#gdcalert7">alert7</a>
<a href="#gdcalert8">alert8</a>
<a href="#gdcalert9">alert9</a>
<a href="#gdcalert10">alert10</a>
<a href="#gdcalert11">alert11</a>
<a href="#gdcalert12">alert12</a>
<a href="#gdcalert13">alert13</a>
<a href="#gdcalert14">alert14</a>
<a href="#gdcalert15">alert15</a>
<a href="#gdcalert16">alert16</a>
<a href="#gdcalert17">alert17</a>
<a href="#gdcalert18">alert18</a>
<a href="#gdcalert19">alert19</a>
<a href="#gdcalert20">alert20</a>
<a href="#gdcalert21">alert21</a>
<a href="#gdcalert22">alert22</a>
<a href="#gdcalert23">alert23</a>
<a href="#gdcalert24">alert24</a>
<a href="#gdcalert25">alert25</a>
<a href="#gdcalert26">alert26</a>
<a href="#gdcalert27">alert27</a>
<a href="#gdcalert28">alert28</a>
<a href="#gdcalert29">alert29</a>
<a href="#gdcalert30">alert30</a>
<a href="#gdcalert31">alert31</a>
<a href="#gdcalert32">alert32</a>
<a href="#gdcalert33">alert33</a>
<a href="#gdcalert34">alert34</a>
<a href="#gdcalert35">alert35</a>
<a href="#gdcalert36">alert36</a>
<a href="#gdcalert37">alert37</a>
<a href="#gdcalert38">alert38</a>
<a href="#gdcalert39">alert39</a>
<a href="#gdcalert40">alert40</a>
<a href="#gdcalert41">alert41</a>
<a href="#gdcalert42">alert42</a>
<a href="#gdcalert43">alert43</a>
<a href="#gdcalert44">alert44</a>
<a href="#gdcalert45">alert45</a>
<a href="#gdcalert46">alert46</a>
<a href="#gdcalert47">alert47</a>
<a href="#gdcalert48">alert48</a>
<a href="#gdcalert49">alert49</a>
<a href="#gdcalert50">alert50</a>
<a href="#gdcalert51">alert51</a>
<a href="#gdcalert52">alert52</a>
<a href="#gdcalert53">alert53</a>
<a href="#gdcalert54">alert54</a>
<a href="#gdcalert55">alert55</a>
<a href="#gdcalert56">alert56</a>
<a href="#gdcalert57">alert57</a>
<a href="#gdcalert58">alert58</a>
<a href="#gdcalert59">alert59</a>
<a href="#gdcalert60">alert60</a>
<a href="#gdcalert61">alert61</a>
<a href="#gdcalert62">alert62</a>
<a href="#gdcalert63">alert63</a>
<a href="#gdcalert64">alert64</a>
<a href="#gdcalert65">alert65</a>
<a href="#gdcalert66">alert66</a>
<a href="#gdcalert67">alert67</a>
<a href="#gdcalert68">alert68</a>
<a href="#gdcalert69">alert69</a>
<a href="#gdcalert70">alert70</a>
<a href="#gdcalert71">alert71</a>
<a href="#gdcalert72">alert72</a>
<a href="#gdcalert73">alert73</a>
<a href="#gdcalert74">alert74</a>
<a href="#gdcalert75">alert75</a>
<a href="#gdcalert76">alert76</a>
<a href="#gdcalert77">alert77</a>
<a href="#gdcalert78">alert78</a>
<a href="#gdcalert79">alert79</a>
<a href="#gdcalert80">alert80</a>
<a href="#gdcalert81">alert81</a>
<a href="#gdcalert82">alert82</a>
<a href="#gdcalert83">alert83</a>
<a href="#gdcalert84">alert84</a>
<a href="#gdcalert85">alert85</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



# NeurIPS 2023 Recap

@rkabra

This is a slice of topics I’ve been interested in of late, including VLMs, scene understanding, object-centric representations, and generative models. The posters below reflect about 2% of NeurIPS. While I have attempted to capture what is trending, there are entire fields that aren’t reflected here. On the whole the conference was less hype-y than last year. There was a lot of focus on evaluation, new ways of using existing models, and generating new data.


### Text-image alignment, attribution, and synthetic data

[DreamSim](https://dreamsim-nights.github.io/): introduces a human-judgment dataset called NIGHTS to capture “mid-level perceptual similarities.” The dataset is collected using a two-alternative forced choice given a reference image. An ensemble of CLIP, OpenCLIP, and DINO is LoRA-tuned on NIGHTS, showing an increase in the DreamSim perceptual metric.



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.jpg "image_tooltip")


Cycle consistency for diffusion models–they derive a set of losses which can be used on unpaired data. Method: they pass in x_0 to enable “reconstruction,” which allows driving away from a given class. 

<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.jpg "image_tooltip")


Attack to alter images imperceptibly so VLMs (MiniGPT-4, BLIP-2) are totally confused when captioning them. Uses a transfer-based attack to maximize similarity of adversarial image, followed by a query-based attack to further maximize similarity between generated caption and target string.



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.jpg "image_tooltip")


Text-image alignment and improved compositional prompting. The DA-Score evaluates alignment using VQA feedback. The proposed Eval-and-Refine method improves alignment and is model-agnostic; it works with Stable Diffusion XL.



<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.jpg "image_tooltip")


[LANCE](https://virajprabhu.github.io/lance-web/): Synthetic image generation by structured prompt variation.



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.jpg "image_tooltip")


“Guided imagination” to expand small datasets. They perturb latent features, optimizing the perturbation to maximize “class informativeness” and a KL-based sample diversity score.



<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.jpg "image_tooltip")


Ask GPT to generate confusing descriptions for better negative examples. They use fine-grained (LLM-generated?) descriptions as true positives. Those aren't verified but are supposedly innocuous. Perhaps there's a risk true positives are wrong and the generated negatives are true? Not convinced how well this can work, but the authors won a CVPR challenge.

<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.jpg "image_tooltip")


Text-image alignment benchmark. Uses an NLI entailment model at Google (Q-squared). (Note to self: PaLI is exposed via an API called Vertex AI.)



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.jpg "image_tooltip")


Diffusion self-guidance showing impressive ability to make compositional corrections in generated images.

<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.jpg "image_tooltip")


Compositional image generation allowing style transfer, negative prompting, etc. New benchmark MCC-250. Image fidelity and text-image alignment measured using FID and CLIP.



<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.jpg "image_tooltip")


First instruction-following VLM called Llava. Helps improve factuality.

<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.jpg "image_tooltip")


An object-centric benchmark to assess generated images for text-image alignment. They study multiple compositional tasks including attribute binding but also counting, position, colors. They show high agreement with human judgment, better than a SoTA CLIPScore. And they show diffusion models are still bad, particularly with position. DeepFloyd IF-XL was the best model (even relative to Stable Diffusion XL).



<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.jpg "image_tooltip")


[UniSim](https://universal-simulator.github.io/unisim/), a video diffusion model which can “simulate realistic experience” of humans/agents interacting with their environment. Can simulate both high-level (e.g., open the drawer) and low-level instructions (e.g., move to x,y). Can be used to train both vision-language planners and low-level RL policies.



<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")


Open-vocabulary part segmentation. They clean and relabel the Pascal-Part and ADE20K-Part datasets.

<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.jpg "image_tooltip")


Quantifying image difficulty based on human viewing time to identify category. They show images to participants for 17ms, 50ms, etc and collect 7 responses. When the majority of participants get it right, that is the MVT. Since this is correlated with VLM performance, do we need to run this exercise again in the future? The authors show CLIP ViT models have the best performance on the hardest images. The authors also mentioned ImageNetX, which comes with labels for lighting and other conditions.



<p id="gdcalert15" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image15.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert16">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image15.jpg "image_tooltip")


Two-object attribute binding: What proportion of feature combinations/”rules” do models need to be trained on to be able to generate the full feature space. They test CNN and ViT based encoders, either using Slot Attention or just a VAE. The hardest version is on ClevrTex.

<p id="gdcalert16" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image16.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert17">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image16.jpg "image_tooltip")


LLMs are great at scoring object-level text-to-image generation!

<p id="gdcalert17" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image17.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert18">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image17.jpg "image_tooltip")


Augmentation to increase dataset diversity while maintaining visual consistency. Utilizes captioning models/LLMs to “extract task-agnostic concepts from training data.” Then augments the training data via language-guided image editing. “To maintain data integrity, a model trained on the original dataset filters out minimal image edits and those which corrupt class-relevant information.”



<p id="gdcalert18" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image18.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert19">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image18.jpg "image_tooltip")


A captioning benchmark/dataset. The authors collect human data using a rating game to achieve community consensus. The aim is to ensure objects, global context, and actions are correctly described in the caption. This produces a caption dataset called VICR. The authors further train a baseline ViLBERT model on VICR and an existing Flickr8k-Expert dataset. VICR training outperforms on all metrics including BLEU, METEOR, ROUGE, and CLIPScore. 

<p id="gdcalert19" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image19.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert20">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image19.jpg "image_tooltip")


Cola: A benchmark to test object attribute binding with hard distractors. Derived from GQA/Visual Genome. The authors advocate multimodal adaptation (i.e., fine-tuning the cross-attention layers) over tuning other parts of the network.



<p id="gdcalert20" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image20.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert21">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image20.jpg "image_tooltip")


Measuring visual perception alignment between humans and models. They take into account cases when humans will abstain from prediction. In uncertain cases, they expect models to imitate human judgements. 



<p id="gdcalert21" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image21.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert22">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image21.jpg "image_tooltip")


Multimodal Information Bottleneck for attribution. Outperforms other methods like GradCAM, Saliency, KernelSHAP, RISE, Chefer et al to attribute CLIP on CC and MS-CXR images. 

<p id="gdcalert22" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image22.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert23">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image22.jpg "image_tooltip")



### 


### Objects/slots/segments

SOLV: Object discovery on real life videos (YouTube) without any additional modalities. They compare with the SoTA, DINOSAUR.



<p id="gdcalert23" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image23.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert24">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image23.jpg "image_tooltip")


VideoSAUR: combines Recurrent Slot Attention with DINOSAUR and adds a temporal similarity loss. Outperforms STEVE on MoVI-E but perhaps slightly worse than SOLV (above)?



<p id="gdcalert24" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image24.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert25">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image24.jpg "image_tooltip")


In-context compositional generation using cross-attention between slots and analogy-based instructions.



<p id="gdcalert25" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image25.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert26">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image25.jpg "image_tooltip")


Cross-attention over slot attention slots can work if diffusing over a pretrained CNN latent space. Plus clustering all slots can produce an interesting concept library (except number of clusters needs to be set).



<p id="gdcalert26" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image26.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert27">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image26.jpg "image_tooltip")


SAMCLR––use SAM segments to sample views for contrastive learning. This helps ensure crops contain sufficient (semantic) overlap.



<p id="gdcalert27" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image27.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert28">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image27.jpg "image_tooltip")


Rotating features––inspiration from neuroscience where magnitude indicates presence of feature while orientation captures semantics. Does not solve BinarizedMNIST. Also did not try spherical features only. 



<p id="gdcalert28" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image28.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert29">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image28.jpg "image_tooltip")


Reusable Slotwise Mechanisms evaluated on CLEVRER and Physion. Compared with SwitchFormer, SlotFormer, and NPS (but also similar to Recurrent Independent Mechanisms?). They emphasize the role of a communication bottleneck between slots.



<p id="gdcalert29" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image29.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert30">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image29.jpg "image_tooltip")


A novel self-supervised framework and model mimicking the mutual exclusivity bias from developmental psychology. Supports multi-object multi-view representation learning, and novel classes (as opposed to object discovery).



<p id="gdcalert30" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image30.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert31">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image30.jpg "image_tooltip")


Explicit symbolic labeling helps generalization on relational composition. The authors predict the next image on a Sticky Shapeworld dataset.



<p id="gdcalert31" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image31.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert32">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image31.jpg "image_tooltip")


A study of occlusions in video action detection. They have 9 severity levels based on background and foreground complexity. They show emergent segregation in capsules.



<p id="gdcalert32" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image32.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert33">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image32.jpg "image_tooltip")



### 


### Contrastive learning

Contextual pretraining using a buffer of data enables semantic segmentation, depth prediction, and in-context decoding via retrieval.



<p id="gdcalert33" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image33.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert34">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image33.jpg "image_tooltip")


FactorCL: Contrastive learning taking into account shared and unique information between X1 and X2. They propose using a conditional InfoNCE lower bound to include information and conditional CLUB upper bound to exclude information.

<p id="gdcalert34" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image34.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert35">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image34.jpg "image_tooltip")


A comparison of video contrastive (MoCo and SimCLR), non-contrastive (BYOL, SimSiam, DINO), generative, and supervised methods under distribution shifts affecting content, viewpoint, actor, etc. Contrastive or Siamese methods learn better viewpoint invariance.

<p id="gdcalert35" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image35.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert36">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image35.jpg "image_tooltip")


Implicit contrastive learning using an asymmetric loss between a source encoder and target encoder (stop-gradient). The asymmetry helps push negative examples apart “in the service of pull”.



<p id="gdcalert36" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image36.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert37">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image36.jpg "image_tooltip")



### Other Theory

[Best paper] Mirage. Emergence by scaling is a phenomenon of tracking nonlinear metrics such as accuracy. With smoother metrics, performance scales smoothly as a function of model size.



<p id="gdcalert37" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image37.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert38">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image37.png "image_tooltip")


Fine-tuning by preserving the pairwise angle of weights rather than additive LoRA. This seems like a more constrained form of fine-tuning, but the authors showed it's enough to fine-tune this way from random weights! The hypersphere energy has been shown to charecterize generalization. They also have new work on language models.



<p id="gdcalert38" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image38.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert39">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image38.jpg "image_tooltip")


Tree of Thoughts implements deliberate problem solving via search in LLMs. 

<p id="gdcalert39" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image39.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert40">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image39.jpg "image_tooltip")


Tree VAEs to enable hierarchical clustering. The tree structure is discovered using an iterative growing schedule.

<p id="gdcalert40" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image40.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert41">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image40.jpg "image_tooltip")


Identifiability theory for nonlinear ICA: what structural sparsity assumptions lead to identifiability of the generative process given independent sources.

<p id="gdcalert41" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image41.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert42">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image41.jpg "image_tooltip")


Causal generalization of ICA.

<p id="gdcalert42" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image42.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert43">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image42.jpg "image_tooltip")


An assessment of the permutation conjecture when pretrained ReLU convolutional filters are flipped (horizontally mirrored). They compare data augmentation and the use of an invariance loss versus the perfect baseline (a graph conv network, GCN). Flipped features are not perfectly equal, suggesting it would make sense to average outputs when permutation equivariance is desired.

<p id="gdcalert43" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image43.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert44">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image43.jpg "image_tooltip")


Quantifying the extent to which Graph NNs model interaction between vertices. This leads to an edge sparsification algorithm with performance benefits.



<p id="gdcalert44" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image44.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert45">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image44.jpg "image_tooltip")


Convolutional State Space Models: help improve long horizon moving MNIST generation over 600 frames. Also evaluated on Minecraft and Habitat long-range benchmarks.



<p id="gdcalert45" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image45.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert46">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image45.jpg "image_tooltip")



### 3D

A 3D feature extractor based on three different methods to inject 3D information into LLMs: features from direct reconstruction (of point cloud and images), grandSLAM, and NeRFs. On the language side, they generate language data using a pipeline based on ChatGPT and BLIP. Evaluation on ScanQA. 

<p id="gdcalert46" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image46.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert47">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image46.jpg "image_tooltip")


[Fusing Stable Diffusion and DINO](https://sd-complements-dino.github.io/) features followed by zero-shot evaluation can outperform SoTA methods on dense and semantic correspondence tasks.

<p id="gdcalert47" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image47.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert48">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image47.jpg "image_tooltip")


Autodecoding to learn latents to diffuse on for 3D voxel diffusion. Evaluation based on FID only because they didn't want to reconstruct objects. Some normalization trick involving the median and IQ distance to make stats of 3D diffusion latents look better. MVImgNet contains real objects but might still be partial views because the objects are placed against something during imaging.



<p id="gdcalert48" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image48.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert49">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image48.jpg "image_tooltip")


sVORF: Slot-guided volumetric radiance fields. (Follow up on uORF?). Impressive segmentation of CLEVR shadows.

<p id="gdcalert49" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image49.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert50">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image49.jpg "image_tooltip")


[Differentiable Blocks World](https://www.tmonnier.com/DBW/): represent shapes from multi-view images using primitive blocks.

<p id="gdcalert50" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image50.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert51">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image50.jpg "image_tooltip")


[PrimDiffusion](https://frozenburning.github.io/projects/primdiffusion/): Diffusion on volumetric primitives to generate 3D humans. Also permits texture transfer (e.g., clothing) and 3D inpainting.



<p id="gdcalert51" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image51.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert52">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image51.jpg "image_tooltip")


Supervising on raw Lidar (photon histograms) is better than simple depth-supervised Nerf. It helps distribute radiance over the full distribution of depth rather than targeting a specific depth. One caveat is all photon data was collected in a dark room.



<p id="gdcalert52" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image52.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert53">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image52.jpg "image_tooltip")


A modular look at 3D-aware image synthesis. They look at point embedding, feature decoding, volume rendering, upsampling, and pose sampling techniques separately. They reproduce models from 2020-2022, showing EG3D was a huge improvement over previous models. They show “volume” point embedding alone (over MLP or Tri-plane representations) gives the best FID on Cats and Cars, while being competitive on FFHQ.

<p id="gdcalert53" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image53.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert54">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image53.jpg "image_tooltip")


OpenMask3D: takes posed RGB-D frames and a reconstructed 3D geometry to produce open-vocabulary 3D instance segmentations. They first get class-agnostic masks, then compute mask features by taking top-k views, and finally use CLIP to query all 3D instances.

<p id="gdcalert54" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image54.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert55">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image54.jpg "image_tooltip")


3D room layout generation from RGB video and “manual 2D annotations of structural elements and their visible parts” (just segmentations?). The method combines point tracking, edge matching, and perpendicularity constraints. The authors release a video and CAD dataset called CAD-Estate.

<p id="gdcalert55" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image55.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert56">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image55.jpg "image_tooltip")


NAVI: a dataset of multi-view and in-the-wild image collections of 36 objects and 10k images. The authors show an improvement over using COLMAP poses. By aligning to 3D, they also provide dense geometric correspondences. 

<p id="gdcalert56" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image56.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert57">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image56.jpg "image_tooltip")


An extension of EPIC Kitchens with 3D geometry (pointclouds and cameras) for 19M frames. Includes VISOR annotations for segmenting objects and hands.

<p id="gdcalert57" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image57.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert58">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image57.jpg "image_tooltip")


Virtual objects pasted into real scenes, with 3D bounding boxes and plausible physical locations. This synthetic data helps achieve SoTA on monocular 3D detection.



<p id="gdcalert58" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image58.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert59">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image58.jpg "image_tooltip")


Planning over partially observed 3D scene graphs using language models. The method PROPHE-C is based on MC trajectory sampling to hallucinate possible 3DSGs.

<p id="gdcalert59" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image59.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert60">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image59.jpg "image_tooltip")


Diffusion over articulation trees (graphs) to generate articulated objects.



<p id="gdcalert60" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image60.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert61">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image60.jpg "image_tooltip")



### 


### Diffusion and GANs (for images)

Text-to-panorama generation using synchronized joint diffusions. They compute the coherence for multiple diffusion processes in _advance_.

<p id="gdcalert61" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image61.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert62">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image61.jpg "image_tooltip")


Diffusion on bounded ranges using beta distributions in both directions. Optimized using KL-divergence upper bounds rather than reweighted ELBOs. They also compare with categorical or count-based diffusion processes.



<p id="gdcalert62" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image62.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert63">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image62.jpg "image_tooltip")


Precision-recall curves for generative models. Each point on Fig 1(b) is a f-divergence that can be optimized. We might choose a specific one based on the quality/diversity trade-off we’re looking for.

<p id="gdcalert63" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image63.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert64">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image63.jpg "image_tooltip")


Particle-based framework to unify GANs and score-based diffusion. They study particle models with and without generators (which enable interaction of particles) versus various gradient vector fields that the generated particles may follow: Wasserstein, Log Ratio, and Discriminator-based.



<p id="gdcalert64" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image64.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert65">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image64.jpg "image_tooltip")


Classification using diffusion models using a labels x timesteps score matrix and timesteps weighting function.

<p id="gdcalert65" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image65.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert66">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image65.jpg "image_tooltip")


Reweighted diffusion losses (which look different from the ELBO) can be seen as ELBO + additive Gaussian data augmentation.

<p id="gdcalert66" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image66.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert67">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image66.jpg "image_tooltip")


Diffusion models for dense vision tasks. Using large-scale synthetic data boosts performance. Evaluation on KITTI flow and NYU depth.

<p id="gdcalert67" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image67.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert68">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image67.jpg "image_tooltip")



### Concepts and disentanglement

Multiple classification taxonomies on CLEVR. MAEs perform the worst on unsupervised clustering (of frozen features) on shape. All models are bad at counting––it’s better to train a ResNet from scratch.



<p id="gdcalert68" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image68.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert69">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image68.jpg "image_tooltip")


Concept algebra as an alternative to prompt tuning to generate images. Every prompt induces a concept distribution which can be mapped to a vector space. Concepts are then modified on their own subspaces.

<p id="gdcalert69" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image69.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert70">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image69.jpg "image_tooltip")


Concepts as dictionary learning and concept importance as attribution to image pixels. Measures like TCAV and Sobol. Visualization of concepts (e.g., what makes an espresso) available [in this demo](https://serre-lab.github.io/Lens/). 

<p id="gdcalert70" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image70.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert71">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image70.jpg "image_tooltip")


[Latest work including James Whittington]. Quantization works better than regularization to learn disentangled latent representations?



<p id="gdcalert71" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image71.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert72">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image71.jpg "image_tooltip")


A theory of compositional generalization. The claims are: it requires a compositional support (i.e., training set supporting all configurations) that is also sufficient (i.e., enables reconstruction of all components).



<p id="gdcalert72" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image72.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert73">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image72.jpg "image_tooltip")



### Fun/novelties/new datasets

Tracking human positions based on the acoustic profile of the room.



<p id="gdcalert73" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image73.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert74">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image73.jpg "image_tooltip")


A fun dataset of chimpanzee actions/behaviors. Permits detection, pose estimation, and spatiotemporal action detection.

<p id="gdcalert74" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image74.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert75">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image74.jpg "image_tooltip")


Efficiency gains in voting unlocked using an explainable randomization step added to deterministic voting rules.

<p id="gdcalert75" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image75.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert76">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image75.jpg "image_tooltip")


Geographically diverse images per object class compared to ImageNet. They identify gaps in model performance (ResNet50, CLIP) from region to region. They also train models on their dataset, and show improvements when testing on a different geo-diverse dataset.

<p id="gdcalert76" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image76.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert77">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image76.jpg "image_tooltip")


DataComp: a large-scale multimodal dataset that outperforms LAION.



<p id="gdcalert77" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image77.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert78">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image77.jpg "image_tooltip")





# Workshops

Object placement in scenes by first hallucinating scenes around objects. A model is trained to calculate placement plausibility based on positive (generated) examples and negative (bad crop) examples. The model can be run on grid points on the target scene to find the best object placement. 



<p id="gdcalert78" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image78.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert79">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image78.jpg "image_tooltip")


DA-Fusion: dreambooth style tokens which allow generating variations of an image for data augmentation. Keeping the token fixed, it is possible to alter the image latent at some stage t of the diffusion generative process.



<p id="gdcalert79" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image79.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert80">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image79.jpg "image_tooltip")


Diffusion models do disentangle when shown only diagonal elements of a feature grid, if they are stopped early during training.



<p id="gdcalert80" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image80.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert81">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image80.jpg "image_tooltip")


SAD: segmentation over RGBD by running SAM on colormap of depth, and then fusing with an OVSeg segmentation of the RGB image.



<p id="gdcalert81" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image81.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert82">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image81.jpg "image_tooltip")


Distance Learner: a new paradigm (same vein as maximum margin?) for classification. Yields out-of-domain understanding because the classifier becomes more uncertain away from the data manifold.

<p id="gdcalert82" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image82.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert83">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image82.jpg "image_tooltip")


Investigating shape bias in ResNets and ViTs. 



<p id="gdcalert83" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image83.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert84">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image83.jpg "image_tooltip")


Synthetic data generation optimized using one-shot feedback from an existing classifier to balance classes. Improves classification performance (in a newly trained model) on underrepresented classes in ImageNet-LT and NICO++.



<p id="gdcalert84" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image84.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert85">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image84.jpg "image_tooltip")


f-GANs settle scores: assuming optimal discriminators, they show the optimal generator must satisfy a score-based equality. The Reverse KL leaves only one term (based on the score) in the product. Better results on mode collapse using ScoreGAN.



<p id="gdcalert85" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image85.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert86">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image85.jpg "image_tooltip")



# 


# Favorite talks

Chelsea Finn (Stanford) at SSL workshop on (1) improving factuality using “semantic entropy,” a model intrinsic quantity, and (2) dealing with cut-off dates using meta-learning (CAMELS) over new information.

Adji Bousso Dieng (Princeton) at SyntheticData4ML on Vendi scores to assess dataset diversity (better than averaging a similarity matrix).

Björn Ommer (LMU, StableDiffusion) keynote on capturing long-range dependencies (e.g., by diffusing on a latent space).

Linda Smith (IndianaU) keynote on how babies might learn long-tailed knowledge from one-shot episodic cues.


# Appendix

If you’re interested, I did a similar recap for ICML 2023 [here](https://docs.google.com/document/d/1MAqNClzFc3M-TF3hsAEBAKsWWq_9HKePHNsr-Onpm-4/edit?resourcekey=0-n0U_jh0FJoiRwCMOF_E3xA&tab=t.0#heading=h.pwkcigpvclvr).
