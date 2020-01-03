# Multilayer Perceptron with Choquet Layers (MLP-CL)

Multilayer Perceptron with Choquet Layers. This work is made on Mex with Matlab. Main features are:

1) Adjustable Exponential Linear Unit (ELU) as hidden layers' activation function

2) Adjustable Hyperbolic Tangent (TANH)  as hidden layers' activation function

3) Adaptive learning rate with cosine annealing warm-restart procedure. It uses validation set's accuracy for deterimining best weights. After epoches finished, you can test your model with test dataset.

4) Weight calculations are programmed with C on Mex like backpropagation.

**NOTE:** `nnInst` is the instance of `nnClass` for this documentation
**NOTE:** You can use `example.m` file for a fresh start.

## VARIABLES
 
**nnInst.nnState:** `train`, `validate` or `test` : `train` includes `validate`

**nnInst.HIDDEN_ALPHA :** `ELU` activation function. set alpha smaller, result is good.
**nnInst.HIDDEN_ALPHA :** `TANH` activation function. alpha=2/x, if we want y==1 for x==1, alpha must be 2.
 
**nnInst.CHOQUET_NODE_SIZE:** `[hidden(1) hidden(2) ... hidden(n)]`
**nnInst.NETWORK_NODE_SIZE:** `[hidden(1) hidden(2) ... hidden(n)]`

**nnInst.ADAP_LEARN_PARAM:** `[MaxLearnRate MinLearnRate EpochPeriod]` , in an epoch period cosine annealing adaptive learning parameter will be applied. for example [0.1 0.001 100] will apply 0.1 to 0.001 learning rate with cosine annealing in each 100 epoch period, if total epoch count is 1000, cosine annealing learning rate will repeat 10 times from 0.1 to 0.001
 
**nnInst.MAX_EPOCH:** maximum epoch count;
 
**nnInst.NORM_COEFF:** `[a b]`; if you call **nnInst.normalizeX(x)**, **x** must be  **x(m,n)** m input nodes n samples.

**nnInst.epochOutputFolder:** Output folder for periodic epoch counts, if it is not determined, there will be no epoch graphics
**epochOutputPeriod:** Period of saving outputs for periodic epoch counts
 
## METHODS


**x(m,n):** input matrix with m input nodes n samples.
**d(k,n):** desired output matrix with k output nodes n samples.

**y(k,n):** output matrix with k output nodes n samples.

**x_:** all outputs of all nodes including input values. `(Multilayer Perceptron Layers)`
**x_diff(i, n):** all derivation outputs of all nodes including input values. `(Multilayer Perceptron Layers)`
**z_:** all outputs of all nodes including input values. `(Choquet Layers)`

	[y, x_diff, x_, z_] = nnInst.choqmlp(x): getting outputs of corresponding inputs x
	nnInst = nnInst.backpropagation(x, d): training outputs of corresponding inputs x


# Publication
Please citate our article if you use this work in your project or publication.

[A new spectral estimation-based feature extraction method for vehicle classification in distributed sensor networks](http://journals.tubitak.gov.tr/elektrik/issues/elk-19-27-2/elk-27-2-33-1807-49.pdf)
**TeX Citation Template:**

    @article{erdem2019new,
      title={A new spectral estimation-based feature extraction method for vehicle classification in distributed sensor networks},
      author={Erdem, KOSE and HOCAOGLU, Ali Koksal},
      journal={Turkish Journal of Electrical Engineering & Computer Sciences}, volume={27}, pages={1120--1131}, year={2019}
    }
