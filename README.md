# Introduction to Quantization
This project was originally concieved as a hackday project for [Deep Hack Mar 2023](https://hasgeek.com/generativeAI/deep-hackathon/), but has since been extended to be a way to introduce people to Quantization. The corresponding talk was done at [GenerativeAI Apr Meetup](https://hasgeek.com/generativeAI/april-meetup/), and the link to it will be updated here when it's done & ready. The deck used in the talk, [is available here](https://github.com/amodm/quantization-intro/blob/main/2023-04-genai-meetup-quantization.pptx)

Updated: 2023-04-24 08:18am IST

## Key Points of the talk

Basics
* Neural Networks are universal function approximators, and as such can _learn_ almost anything
* What they learn gets stored as matrices (technically, tensors) of _weights_ & _biases_. Both of these are usually floating point numbers.
* Floating point (FP) numbers can be represented in computers at different precisions (and thus taking different memory space), e.g. 32-bit floating point number is more precise than a 16-bit number.
* GPUs are faster at matrix operations, and so can _process_ neural networks much faster, but this requires the model to fit in GPU memory, as system memory access is much slower. Smaller (less precise) representations of FP numbers can make this possible.
* Smaller representations of FP numbers also require lesser compute, so GPU can execute more such operations per cycle, a win-win.
* Quantization involves converting bigger FP representations to smaller, to reduce model size, so that it can fit in memory and run faster.

Demo
* Despite reducing precision significantly, we see that performance of the neural network largely remains the same during inference. This shows that DNNs are very resilient to small perturbations in the weights & biases.
* How can we leverage this to quantize better?

Uniform Quantization (INT8)
* We can scale a range of FP numbers (say α to β) to 8-bit integers (-127 to 127) uniformly (technique detailed in talk), by projecting it to the -127 to 127 range.
* Quantization in this way will obviously cluster a bunch of FP numbers to the same INT8 number.
* Dequantization is a straightforward reversal of the process, with the caveat that we don't recover the original number, we recover only the center of the range that got projected to this INT8 number.
* The talk describes the _absmax_ variant of it (i.e. taking the max of abs value of α,β and using that as input range to be projected). There's a _zeropoint_ variant as well, which is not described here.
* Similarly, we can choose to project to INT4 (4-bit integers) than INT8.

Distribution of weights:
* FP numbers in the weights matrix can be distributed in different ways. If we're not careful, a large number of weights can map to the same bucket during quantization, reducing accuracy & perplexity.
* Different techniques use different ways to handle these distributions. For simple ones, _absmax_ INT projection (described above) works, but not always.
* Doing INT projection per row/col of matrix is very often used (some nuances to this that aren't described here).
* LLM.int8 paper shows that at larger scales, outlier features show up which greatly affect output (and so cannot be pruned). We cannot use the row/col quantization, because feature dimensions (axis) lie orthogonal to the matrix product dimensions. LLM.int8 paper presents a method to handle them.
* SmoothQuant takes a different approach to the above problem by _smoothening_ the spike in input (X) by scaling it down with a factor, while scaling up the appropriate weight matrix (Y) entries correspondingly. This effectively _shifts_ the spikes from X to Y, giving us a better range to work with during quantization.

We ran out of time to cover other slides in the talk, but you can refer to some minor comments in the presented notes of those slides.

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
