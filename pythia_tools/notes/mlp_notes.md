# Notes on MLPs in GPT models

This text consists of notes of conceptual and empirical investigations of the multilayer perceptrons in causal transformer language models, specifically models from the Pythia suite. The goal is to collect the basic empirical knowledge to reason about their function and role in the network and find the most useful approaches to interpreting the specific MLPs in specific models. 

The notes will contain links to Google Colab notebooks that provide the evidence for the empirical claims. Some of the code for making these investigations is wrapped up in a Python package https://github.com/kmrasmussen/pythia_tools

## What is the MLP?
The code for Pythia models can be found here: https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L566
A Pythia model is structured as follows:
* Embedding $v \times d$ matrix where $v$ is vocabulary size (the number of tokens) and $d$ is the residual stream dimensionality. There is no affine layer
* $L$ transformer blocks, we refer to a block using zero indexing
  * A multi-head self-attention providing an additive update to the residual stream. LayerNorm is applied to the input.
  * An MLP that takes the input from the residual stream after the attention part:
    * LayerNorm of the input vector $x \in R^d$
      * $LayerNorm(x) = \frac{x - E[x]}{\sqrt{Var(x)}}* a_{ln} + b_{ln}$, where $a_{ln}$ and $b_{ln}$ are vectors in $R^d$ called scale and shift vectors respectively.
    * An in-affine: $W_{in}$ is a $4d \times d$ matrix and a bias vector $b_{in}$ of dimensionality $4d$.
    * GELU activation function applied to each entry in the 4d output of the in-affine.
    * An out-affine: $W_{out}$ is a $d \times 4d$ matrix and a bias vector $b_{out}$ of dimensionality $d$.
* Unembedding $v \times d$ matrix

### One possible conceptual view of the MLP
We say that in an MLP there are $4d$ neurons that activate according to the GELU activation function. For neuron $i$ there is an associated row vector in $W_{in}$ and an associated column vector in $W_{out}$, both of dimensionality $d$.
* We call the row vector the receptor: It is a vector in $R^d$ that is placed so that the neuron fires when the LayerNormalized input vector has a high dot-product with its receptor. If the input vector and the receptor vector were both unit-norm this would be the cosine similarity between them. We will look empirically at the norms of inputs and receptors in the following sections.
  * Due to the LayerNorm, it might not be useful to think of receptors and inputs as living in residual stream space. However, it might be worth somehow looking at how much this is the case.
  * A general question for an MLP is how the unit-normalized receptors are distributed on the hypersphere. Are they roughly evenly distributed? Are there some receptors which are far apart from others?
  * The in-bias $b_{in,i}$ is a scalar value and is the pre-activation of the neuron.
* We will call the column vector the value vector: To the degree that the input is dot-product close to the receptor, the neuron will fire, and to the degree it fires the neurons value vector is written to the residual stream. The scaled value vector plus its out-bias $b_{out,i}$ we will call an MLP subupdate.
  * Note that each neuron is acting separately, the output written to the residual stream by the MLP is the sum of subupdates plus the out-bias.
  * Note that the in-bias and out-bias have quite different interpretations in this conceptual view: The in-bias consists of pre-activations for each neuron, while the out-bias is a d-dimensional vector that is "global" to the MLP.

## Basic statistics of MLPs

It is often useful to frame investigations as being either static or non-static: By static investigation we mean that we look at a loaded trained Pythia model and just looking at parameters without making any forward passes. By non-static investigation we mean that we take some data, in our case mostly a small subset of the validation set (so "on-distribution data") and look at how the model behaves in this case. The most basic static investigation is to look at simple statistics and visualizations of the parameters.

### Norms of row vectors
We go through all MLPs of all the final Pythia models and compute the L2 norm of the row vectors. For a layer we take all the norms and make a histogram and boxplot to get a sense of the distribution.
**TODO: Insert empirical findings and plots**
Inside a single model, there is some variation among the distribution of norms for a layer. In model 1B it is roughly the case that for most layers the distribution is centered at norm 1 with most of the mass centered in the interval (0.8,1.2)
**TODO: Compute means and variances and medians for each layer**

### Norms of column vectors
**TODO: Insert empirical findings and plots**

### In-biases
Colab https://colab.research.google.com/drive/1wJRmVwaXC8ECGPghax2JcJWhs2qcyywY?usp=sharing

*How are entries distributed in the in-bias?* Since the in-bias $b_{in}$ is interpreted as the pre-activation, when looking at a specific MLP it is worth looking at how the entries in its $b_{in}$ is distributed. Histograms and boxplots for each layer in each model can be found here https://antipiano.com/pythia_browser/section/in_bias/
We see that the biases are in general slightly negative around -0.1. In 1B the median is closer to 0. There are some cases of outliers with very high bias terms. This is especially true in 70m where layer 4 and 5 have neurons around 1. In 1B the highest biases are not so large, not much larger than 0.1, except for the first two layers where some neurons are > 0.3.

### Activation frequencies
*How often are MLP neurons active?* A ReLU neuron is active when its receptor dotted with the input plus its bias is greater than 0. The GELU has a bit more complicated shape, but the idea should still be useful. In a non-static investigation we can take N sequences of length T and feed through the model, and look at the $NT$ different activation in a specific layer. For neuron $i$, how large a fraction of the $NT$ cases resulted in positive activation, the activation-fraction of the neuron. How are activation-fractions distributed in a layer? Since we are looking at all the T activation vectors for each sequence there might be a problem that activations are correlated within a sequence, however this might not be such a big problem. Alternatively one can take a larger $N$ and sample a random position to address this concern.



# Other


For a neuron, there is an associated row vector, we call this vector the receptor: The neuron will fire if the input has a high dot product with the receptor. With respect to a specific receptor, we will think of it as the north pole, and for ReLU the south pole represents the neuron not firing.

## How often are neurons active?
A basic question that is worth answering in order to better understand MLPs is how often neurons are active. There are many ways to understand this question and ways to answer these questions. We are using Pythia models and as data we use 10K sequences of length 600 from the Pile validation set.

* How do we define whether a neuron is active? For a start, we will say that a neuron is active if the input to GELU is greater than 0.
* For a single sequence, model, layer combination:
    * For each position what is the fraction of neurons that activate? What does the distribution of these values look like?
    * For each neuron what is the fraction of positions in the sequence where it activates? What does the distribution of these values look like?
* For multiple sequences and fixed model and layer:
    * Lumping together positions for different sequences into a set of positions - for each neuron what is the fraction of positions in the sequence where it activates?
* It seems that there are certain neurons whose activation fraction is outlier high. For example in 410m-layer 4 the neuron 3364 is active in .7 of tokens. However the row vector for neuron 3396 is among the very lowest in terms of norm, which is also relevant, but does not directly explain it.
* Conceptually there is something counter-intuitive about the distribution of fractions of inputs that makes neurons fire: For example, for half of the receptors in layer 4, only 12% of inputs are on its north pole. Neuron 410-20-2443 has an active-fraction of .95. This means that .95 of inputs are on the same half. This neuron is not Neuroscope-interpretable.
  * The pattern is not so simple, 410-11-909 has the highest activation-fraction in 410-11 and is Neuroscope-interpretable as "launching software when described in Greek". This feature is of course very rare in the data. 410-11-1512 is almost monosemantic with some weird things in between as "the word turn in the sense of something turning into something else"



# Questions
* Are there neurons where fixing its activation to be high results in visibly different generated text?
* Feature visualization is easier for vision models because one can directly compute the derivative of the activation wrt to the input. 
    * What happens if you directly optimize the post-embedding representations for a fixed length sequence and then after finding maximal activation you try to look at which tokens it is nearby, or which text is generated by continuing this text?
    * Alternatively, the input sentence which is supposed to maximally activate the neuron could be generated by the model itself. What happens if you generate using a model but you don't fix a specific predicted token, you just use the distribution over tokens and make a weighted average of the embedding tokens to continue the text. In this way the discrete choice does not kill the differentiability, but probably still to compute intensive.
* If a neuron is Neuroscope-interpretable, and you have a clear idea, e.g. 410-11-3645 seems to be something like "pursuing a long travel (by some vehicle)" - should we then expect another model to be able to generate texts that significantly activate that neuron? If it does not, what is the implication wrt to the degree to which we have an interpretation for the neuron?

# Superposition
* Is there any way to estimate the number of virtual neurons contained in an MLP? This is related to the amount of noise tolerated, so if there is an indirect way of estimating the noise levels - which level of intereference is tolerated it could be used.
  * One can find column vectors where top-k has this pattern: Column vector A top-k looks quite clean, Column vector B top-k looks quite clean but contains tokens from vector A top-k suggesting that the feature that was thought to be registered by neuron A is actually registered by a virtual neuron somewhere more in the direction of B.
    * If this idea makes sense, would that not suggest that look at statistics over the dataset of how receptor A and receptor B works? And when this is observed, is receptor B significantly close to A?
  * Is there intereference at the level of deembedding? If superposition is sufficiently useful, then the model might sacrifice a bit higher probability of the tokens it actually thinks are the next ones for having another "mode"/set of tokens, that it really does not think is likely, but because of the training objective it will not hurt - it is only indirectly encouraged to minimize the probability for these. Is there some simple way of looking at whether stuff like this exists? Looking through top-k predictions for quite large k and seeing if there is superposition there
* 

