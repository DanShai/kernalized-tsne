# Kernalized tsne

A simple alternative implementation of Laurens van der Maaten t-Distributed Stochastic Neighbor Embedding (t-SNE) , with use of kernals.

# equations

- joint probability pij
  ![Screenshot](img/pij.png)
- joint probability qij
  ![Screenshot](img/qij.png)
- Cost
  ![Screenshot](img/cost.png)
- gradient
  ![Screenshot](img/gra.png)

- gradient update
  ![Screenshot](img/graupdate.png)

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
