
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ktsne import Ktsne
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


iris = datasets.load_iris()
X = iris.data
y = iris.target
scaler = MinMaxScaler(feature_range=(0, 1))

f_opts = {'p_degree': 2.0, 'p_dims': 3, 'eta': 10.0,
          'perplexity': 30.0, 'n_dims': 2, 'ker': 'poly', 'gamma': 1.0}
k_tsne = Ktsne(X, f_opts=f_opts)

kernel = f_opts["ker"]

plt.clf()


X1 = scaler.fit_transform(X)
x_min, x_max = X1[:, 0].min() - .5, X1[:, 0].max() + .5
y_min, y_max = X1[:, 1].min() - .5, X1[:, 1].max() + .5

plt.subplot(2, 1, 1)
# Plot the training points
plt.scatter(X1[:, 0], X1[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("without ktsne")


X_reduced = k_tsne.get_solution(3000)
X_reduced = scaler.fit_transform(X_reduced)
plt.subplot(2, 1, 2)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel(' V1 ')
plt.ylabel(' V2 ')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("with ktsne %s kernel " % kernel)


plt.show()
