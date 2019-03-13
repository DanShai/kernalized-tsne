# Kernalized tsne

A simple alternative implementation of Laurens van der Maaten t-Distributed Stochastic Neighbor Embedding (t-SNE) , with use of kernals.

#equations

- similarities gaussian conditional probabilities:

$`p_{j\vert i}=\frac{\exp(-\| x_i-x_j\|^2/2\sigma^2)}{\sum_{k\neq i}\exp(-\| x_i-x_k\|^2/2\sigma^2)}`$

- output similarities t-distribution conditional probabilities:

```math
q_{ij}=\frac{(1+\| y_i-y_j\|^2)^{-1}}{\sum_{k\neq l}(1+\| y_i-y_l\|^2)^{-1}}
```

- gradient:

```math
\frac{\delta C}{\delta y_i}=4\sum_j(p_{ij}-q_{ij})(y_i-y_j)(1+\| y_i-y_j\|^2)^{-1}
```

for more information read the PDF file included.

Apart from pca, available kernels are:

- iquad
- cauchy
- fourier
- rbf
- poly
- cosine
- anova

you can add new/own kernel and test it, make sure to tune the parameters in f_opts to get better results!

License : MIT and you are free to do what ever you want with it !

![Screenshot](/ktsne.png)
![Screenshot](/ktsne2.png)
