# Notes on MLPs in GPT models

This text consists of notes of conceptual and empirical investigations of the multilayer perceptrons in causal transformer language models, specifically models from the Pythia suite. The goal is to collect the basic empirical knowledge to reason about their function and role in the network and find the most useful approaches to interpreting the specific MLPs in specific models. 

The notes will contain links to Google Colab notebooks that provide the evidence for the empirical claims. Some of the code for making these investigations is wrapped up in a Python package https://github.com/kmrasmussen/pythia_tools. Some of the data produced in empirical investigations are placed on a website https://antipiano.com/pythia_browser/, notebooks to produce the data for Pythia browser is here https://github.com/kmrasmussen/pythia_browser

## What is the MLP?
The code for Pythia models can be found here: https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L566
A Pythia model is structured as follows:
* Embedding $v \times d$ matrix where $v$ is vocabulary size (the number of tokens) and $d$ is the residual stream dimensionality. There is no bias.
* $L$ transformer blocks. I refer to a block using zero indexing.
  * A multi-head self-attention providing an additive update to the residual stream of each position. LayerNorm is applied to the input.
  * An MLP that is applied to each residual stream separately to give additive updates to each residual stream separately. Itakes the input from the residual stream after the attention part:
    * LayerNorm of the input vector $x \in R^d$
      * $LayerNorm(x) = \frac{x - E[x]}{\sqrt{Var(x)}}* a_{ln} + b_{ln}$, where $a_{ln}$ and $b_{ln}$ are vectors in $R^d$ called scale and shift vectors respectively.
    * An in-affine: $W_{in}$ is a $4d \times d$ matrix and a bias vector $b_{in}$ of dimensionality $4d$.
    * GELU activation function applied to each entry in the 4d output of the in-affine.
    * An out-affine: $W_{out}$ is a $d \times 4d$ matrix and a bias vector $b_{out}$ of dimensionality $d$.
* Unembedding $v \times d$ matrix

### One possible conceptual view of the MLP
We say that in an MLP there are $4d$ neurons that activate according to the GELU activation function. For neuron $i$ there is an associated row vector in $W_{in}$ and an associated column vector in $W_{out}$, both of dimensionality $d$.
* We call the row vector the receptor: It is a vector in $R^d$ that is placed so that the neuron fires when the LayerNormalized input vector has a high dot-product with its receptor - but bias is included. If the input vector and the receptor vector were both unit-norm this would be the cosine similarity between them. We will look empirically at the norms of inputs and receptors in the following sections.
  * Due to the LayerNorm, it might not be useful to think of receptors and inputs as living in residual stream space. However, it might be worth somehow looking at how much this is the case.
  * A general question for an MLP is how the unit-normalized receptors are distributed on the hypersphere. Are they roughly evenly distributed? Are there some receptors which are far apart from others?
  * The in-bias $b_{in,i}$ is a scalar value and is the pre-activation of the neuron.
* We will call the column vector the value vector: To the degree that the input is dot-product close to the receptor, the neuron will fire, and to the degree it fires the neurons value vector is written to the residual stream. The scaled value vector we will call the neurons subupdate.
  * Note that each neuron is acting separately, the output written to the residual stream by the MLP is the sum of subupdates plus the out-bias. The sum including the bias we will call the total update.
  * Note that the in-bias and out-bias have quite different interpretations in this conceptual view: The in-bias consists of pre-activations for each neuron, while the out-bias is a d-dimensional vector that is "global" to the MLP.


## Basic statistics of MLPs

It is often useful to frame investigations as being either static or non-static: By static investigation we mean that we look at a loaded trained Pythia model and just looking at parameters without making any forward passes. By non-static investigation we mean that we take some data, in our case mostly a small subset of the validation set (so "on-distribution data") and look at how the model behaves in this case. The most basic static investigation is to look at simple statistics and visualizations of the parameters.

### Norms of row vectors
We go through all MLPs of all the final Pythia models and compute the L2 norm of the row vectors. For a layer we take all the norms and make a histogram and boxplot to get a sense of the distribution.

The plots can be found here https://antipiano.com/pythia_browser/section/in_row_norm/

Inside a single model, there is some variation among the distribution of norms for a layer. In model 1B it is roughly the case that for most layers the distribution is centered at norm 1 with most of the mass centered in the interval (0.8,1.2)
**TODO: Compute means and variances and medians for each layer**

Overall the plots do not give any very clear or useful conclusions. Later we will present the idea of folding in layernorm into the in-affine, which means will lead to a new in-affine. These rows can be interpreted separately.

### Norms of column vectors
We do the same for column vectors. Here we do not have the option of an alternative view based on folding in layer-norm. 

The plots can be found here https://antipiano.com/pythia_browser/section/out_col_norm/

The shape of the distributions are most clear for larger models, 410m and 1b

Observations:
* For 410 the most layers have distributions centered around 0.6 with a right-tail of outliers around 1.6. For 1b the distributions are centered around 0.8 wiht right outliers above 2
* Some distributions are more skewed than others, is there a pattern with early and late layers being more symmetric?

### In-biases

*How are entries distributed in the in-bias?* Since the in-bias $b_{in}$ is interpreted as the pre-activation, when looking at a specific MLP it is worth looking at how the entries in its $b_{in}$ is distributed. Histograms and boxplots for each layer in each model can be found here https://antipiano.com/pythia_browser/section/in_bias/
We see that the biases are in general slightly negative around -0.1. In 1B the median is closer to 0. There are some cases of outliers with very high bias terms. This is especially true in 70m where layer 4 and 5 have neurons around 1. In 1B the highest biases are not so large, not much larger than 0.1, except for the first two layers where some neurons are > 0.3.

### Activation frequencies
*How often are MLP neurons active?* A ReLU neuron is active when its receptor dotted with the input plus its bias is greater than 0. The GELU has a bit more complicated shape, but the idea should still be useful. In a non-static investigation we can take N sequences of length T and feed through the model, and look at the $NT$ different activation in a specific layer. For neuron $i$, how large a fraction of the $NT$ cases resulted in positive activation, the activation-fraction of the neuron. How are activation-fractions distributed in a layer? Since we are looking at all the T activation vectors for each sequence there might be a problem that activations are correlated within a sequence, however this might not be such a big problem. Alternatively one can take a larger $N$ and sample a random position to address this concern. The resulting histogram from using this alternative approach looks very much like the hisogram from the first approach. The first approach requires less compute, since it has fewer forward passes.

Histograms and boxplots for each layer in each model (up to 410m) can be found here
https://antipiano.com/pythia_browser/section/act_frac/. We used the method where we use all activations for all tokens for all sequences, but we only use 10 sequences. Therefore one should not draw too strong conclusions from the data.

We draw two conclusions from these plots:
* Across layers and models, it is common for neurons to have an activation-fraction of .2.
* In all layers and models, there are outlier neurons that have activation-fractions above .8.



### Folding in LayerNorm
(The folding in of LayerNorm presented in this section is very related to the folding in in TransformerLens, but TransformerLens does not fold in the constant that turns standardization into L2-normalization.)

For fixed layernorm parameters $a_{ln}$, $b_{ln}$ and in-affine with $W_{in}$, $b_{in}$, one can fold the layernorm and in-affine into each other in a way that can be conceptually useful:

In this section $x$ is the input to layernorm, not the input to the in-affine., i.e. $x$ is the vector read from the residual stream. First, layernorm the shift and scale operation in layernorm is an affine transformation, because the scaling can be written as a diagonal matrix $A_{ln}$ where the diagonal is the vector $a_{ln}$ The demeaning is a linear transformation, can be written as a matrix $D$ with diagonal entries being $1-1/d$ and the off-entries being $1/d$. Dividing by the standard deviation can be written as dividing by the product of $\sqrt{1/(d-1)}$ and the L2 norm of the demeaned vector.

$$
LayerNorm(x) = A_{ln}(\sqrt{\frac{1}{d-1}} \frac{Dx}{||Dx||}) + b_{ln}
$$
where the scaling $\sqrt{\frac{1}{d-1}}$ is a linear operation $S$, meaning we can make $A'_{ln} = A_{ln} S$. Let $x' = \frac{Dx}{||Dx||}$ then LayerNorm is
$$
LayerNorm(x) = A'_{ln}x' + b_{ln}
$$
This input is fed to the MLP-in affine, so
$$
MLP_{in}(x) = W_{in}(A'_{ln}x' + b_{ln}) + b_{in} = (W_{in}A_{ln}')x' + (W_{in}b_{ln} + b_{in}) = W'_{in}x' + b'_{in}
$$
This I will call the folded in-affine and I will call the space where $x'$ lives the in folded receptor space.

This suggests a view of the LayerNorm-MLP block that is like this: $x$ is taken from the residual stream and moved by $D$ to folded-receptor-space where it is then normalized to $x'$. In folded receptor space, inputs are always unit norm. 

Since inputs $x'$ are unit-norm, neuron activations can be understood in terms of cosine similarity. A neuron with receptor $r$ and bias $b$ has activation when:
$$
r * x' + b = ||r||sim(r,x') + b > 0 \implies sim(r,x') > \frac{-b}{||r||}
$$

Consider $r$ fixed with a direction and norm. Then it could make sense to define $\epsilon_r = \frac{-b}{||r||}$ and look at the probability that a random input $X$ selected uniformly from the hypersphere where inputs in folded receptor space live and looking at the probability that $sim(r,X) > \epsilon_r$. We could call it the activation volume for the receptor.

Though the models use GELU, consider for now ReLU neurons. The neuron activation is then
$$ReLU(||r||sim(r,x') + b)
$$
which can be viewed first as a shifted ReLU which only starts activating at $-b$ and then does so, not with slope 1 but with slope $||r||$
$$ReLU_{-b}^{||r||}(sim(r,x'))$$
I'm not sure about this, but this seems to suggest that for ReLU neurons, it is not active outside the activation volume and then activation increases linearly. Notice the upper bound on the activation is then.
## Static investigation: Activation volumes
**TODO: Implement the folding in as described in the above section and compute (approximate using uniforom sampling) the activation volumes for neurons and plot histograms and boxplots**

## Non-static investigation: How many neurons are involved in a total update?
We have defined the notions of subupdate and total update above. Above we saw that often a neuron has an activation fraction of .2. A related question is, for a specific total update (i.e. the output of a specific MLP at a specific position in a specific sequence), how many neurons are active, and more generally how much do the subupdates from each neuron contribute to the total update.

For the first case we take 10 sequences of length 600, and for each model and layer we look at the $10 \cdot 600$ activation vectors of size $4d$ (after the GELU). How many are positive, that is how many neurons are "active", how large a fraction?

Histogram and boxplot and boxplot can be found here https://antipiano.com/pythia_browser/section/token_act_frac/

We find that using 10 sequences is enough to get a general feel for the shape, in the sense that using another 10 sequences will give roughly the same shape.

If all neurons activated iid probability $p$, then the expected of neurons active is also $p$. From the plots we see that the generally the distributions of neuron activation fraction and token activation fraction have peak at roughly the same place which is roughly $.2$.

There are many cases where outliers in token activation fraction distributions form a small mode above .5

## How much do subupdates contribute to the total update?
When investigating this we should remember that GELU can take on negative values. In many cases a very large fraction of activations will be <-0.1. This cannot be ignored. If you remove the subupdates for which the activation is negative the resulting update will have almost no cosine similarity to the total update (cosine sim < 0.03)

If we take all the total updates without the bias (so the sum of the subupdates) in a sequence and take their mean, we can call it the subupdate mean. For some layers the subupdate means are very cosine-similar across sequences. In the cases where they are very similar we can use a fuzzy concept of a virtual bias. I hypothesize that this virtual bias is explained by neurons with high bias terms which are always active with relatively high activation. It might be necessary to reframe subupdates where this virtual bias is factored out. How does one remove this virtual bias? One way could be to compute the mean subupdate for each 

One way of looking at contributions to total subupdates is to make cumulative sums of subupdates add the bias and compare to the total sum of subupdates plus bias which is the total update in term of cosine similarity. One can also do this without the bias. One way of ordering the sums is by their activation fraction. Another is by the norm of the updates.

Ordered by descending activation the pattern is one of rapid increase in the cosine similarity using the first hundreds of highest ranking subupdates up to 0.6-0.7 followed by a plateau and then a slow rise to similarity 1. This curve is not monotonically increasing. It will be worth 

# MLP input-output Jacobians
The MLP is a function $f : R^d \to R^d$. For any input $x \in R^d$, the Jacobian $J_f(x)$ is a matrix of all the partial derivatives.


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

