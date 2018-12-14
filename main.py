
import matplotlib.pyplot as plt

from ktsne import Ktsne
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

digits = datasets.load_digits()
X = digits.data
y = digits.target


X, y = shuffle(X, y)
X = X[:500]
y = y[:500]

scaler = MinMaxScaler(feature_range=(-1, 1))

f_opts = {'p_degree': 2.0, 'p_dims': 12, 'eta': 25.0,
          'perplexity': 50.0, 'n_dims': 2, 'ker': 'pca', 'gamma': 1.0}

kernel = f_opts["ker"]

plt.clf()


X1 = scaler.fit_transform(X)

plt.subplot(2, 1, 1)
# Plot the training points
X_pca = PCA(n_components=2).fit_transform(X1)
x_min, x_max = X_pca[:, 0].min() - .5, X_pca[:, 0].max() + .5
y_min, y_max = X_pca[:, 1].min() - .5, X_pca[:, 1].max() + .5

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('p1')
plt.ylabel('p2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title(" PCA without ktsne ")


k_tsne = Ktsne(X1, f_opts=f_opts)
X_reduced = k_tsne.get_solution(3000)
X_reduced = scaler.fit_transform(X_reduced)
plt.subplot(2, 1, 2)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
x1_min, x1_max = X_reduced[:, 0].min() - .5, X_reduced[:, 0].max() + .5
y1_min, y1_max = X_reduced[:, 1].min() - .5, X_reduced[:, 1].max() + .5

plt.xlabel(' V1 ')
plt.ylabel(' V2 ')

plt.xlim(x1_min, x1_max)
plt.ylim(y1_min, y1_max)
plt.xticks(())
plt.yticks(())
plt.title("with ktsne %s kernel " % kernel)


plt.subplots_adjust(hspace=0.5)
plt.show()
