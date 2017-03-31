import numpy as np
import matplotlib
# matplotlib.use('MACOSX')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = np.genfromtxt('input3.csv', delimiter=',', skip_header=1)
dataset.shape
X = dataset[:, 0:2]
X.shape
y = dataset[:, -1]
y.shape
y[0:10]
dataset[0:10, :]
X[:, 0].mean(), X[:, 0].std()
X[:, 1].mean(), X[:, 1].std()
fig, ax = plt.subplots()

plt.ioff()
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='+', c=y[y == 0], cmap=cmap_light)
ax.set_xlabel('A')
ax.set_ylabel('B')

ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c=y[y == 1], cmap=cmap_bold)

# plt.show()
from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
h = 1  # step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

# plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3,levels=np.linspace(0, 1, xx.shape[0]))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3, levels=np.linspace(0, 1, Z.shape[0]))

plt.xlim(X[:, 0].min(), X[:, 0].max())
plt.ylim(X[:, 1].min(), X[:, 1].max())


# plt.show()
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
