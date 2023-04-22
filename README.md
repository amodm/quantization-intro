# Introduction to Quantization
This project was originally concieved as a hackday project for [Deep Hack Mar 2023](https://hasgeek.com/generativeAI/deep-hackathon/), but has since been extended to be a way to introduce people to Quantization. The corresponding talk was done at [GenerativeAI Apr Meetup](https://hasgeek.com/generativeAI/april-meetup/), and the link to it will be updated here when it's done & ready. The deck used in the talk, [is available here](https://github.com/amodm/quantization-intro/blob/main/2023-04-genai-meetup-quantization.pptx)

Updated: 2023-04-22 10:55pm IST

## Resources
* Links included in the slides
  * Slide 1: This repo
  * Slide 2: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - running quantized LLaMA models on a MacBook
  * Slide 6: [An Image is worth 16x16 words](https://arxiv.org/abs/2010.11929) - ViT (Vision Transformer) paper
  * Slide 6: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
  * Slide 8: [Nvidia Hopper Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
  * Slide 10: [IEEE-754 Playground](https://www.h-schmidt.net/FloatConverter/IEEE754.html)
  * Slide 10: [FP formats used in the DNN world](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407)
  * Slide 11: [A survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
  * Slide 15: [LLM.int8() paper](https://arxiv.org/abs/2208.07339)
  * Slide 16: [LLM.int8() paper](https://arxiv.org/abs/2208.07339)
  * Slide 17: [SmoothQuant paper](https://arxiv.org/abs/2211.10438)
  * Slide 18: [The case for 4-bit precision: k-bit scaling laws](https://arxiv.org/abs/2212.09720)
  * Slide 20: [GPTQ paper](https://arxiv.org/abs/2210.17323)
  * Slide 21: [A survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
* Over-parametrization related
  * [Predicting Parameters in Deep Learning](https://arxiv.org/abs/1306.0543)
  * [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933)
  * [Demystifying over-parametrization in DNNs](http://www.ipam.ucla.edu/abstract/?tid=15453&pcode=GLWS3)
* Other papers:
  * [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568) - AdaRound paper
  * [Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming](https://arxiv.org/abs/2006.10518) - AdaQuant paper - A layer-by-layer optimization method that minimizes the error between the quantized layer output and the full-precision layer output.
  * [ZeroQuant paper](https://arxiv.org/abs/2206.01861)
  * [Optimal Brain Compression](https://arxiv.org/abs/2208.11580)
  * [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)

## The Demo - Studying the effect of reduced precision
In this demo, we use [@hila-chefer](https://github.com/hila-chefer)'s work on [Transformer Explainability](https://github.com/hila-chefer/Transformer-Explainability) to empirically study the effect of reduced precision at inference time. The rather successful recent efforts to deploy [llama.cpp](https://github.com/ggerganov/llama.cpp) on hardware as weak as [a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) was a major factor to explore this.

### Approach
[IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) standard for floating point precision describes a float32 as `1-bit (sign); 8-bits (exponent); 23-bits (fraction)`. We take a rather crude approach to simulating precision by simply truncating the least `n` significant bits of the fractional part to zero. We do this for all the parameters of the model.

Important caveats:
    * While the model's precision was varied, computations were still being done as float32.
    * Exponent part was not touched, so the range of the model weights remained the same.

### How to use
* `pip install -r requirements.txt`
* `streamlit run app.py`

### Results
For every image we tested, the model was able to be resilient at inference time to about 3-bits of fractional precision (translating roughly to a single decimal point).
