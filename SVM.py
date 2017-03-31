
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

import matplotlib


# In[3]:

#matplotlib.use('MACOSX')


# In[4]:

import matplotlib.pyplot as plt


# In[5]:

from matplotlib.colors import ListedColormap


# In[6]:

dataset = np.genfromtxt('input3.csv', delimiter=',', skip_header=1)


# In[7]:

X = dataset[:, 0:2]


# In[8]:

y = dataset[:,-1]


# In[9]:

fig,ax = plt.subplots()


# In[10]:

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAcAFF'])


# In[11]:

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# In[12]:

ax.scatter(X[y == 0 ,0], X[y == 0, 1], marker='+', c=y[y==0], cmap=cmap_light)


# In[13]:

ax.set_xlabel('A')


# In[14]:

ax.set_ylabel('B')


# In[15]:

ax.scatter(X[y == 1 ,0], X[y == 1, 1], marker='o',c=y[y==1],cmap=cmap_bold)


# In[16]:

#plt.show()


# In[17]:

from sklearn import svm


# In[18]:

from sklearn.model_selection import train_test_split


# In[19]:

from sklearn.model_selection import cross_val_score


# In[44]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)


# In[45]:

C = 0.5


# In[46]:

kernel = 'linear'


# In[47]:

gammas = np.logspace(-6, -1, 10)


# In[48]:

clf = svm.SVC(C=C)


# In[49]:

#cross_val_score(clf, X_train, y_train, cv=5)


# In[40]:

#clf.fit(X_train, y_train)


# In[50]:

#clf.score(X_test, y_test)


# In[51]:

#from sklearn.model_selection import ShuffleSplit


# In[52]:

cv=5


# In[74]:

param_grid = [
  {'C': [1, 10, 100], 'kernel': ['linear']},
  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


# In[75]:

from sklearn.model_selection import GridSearchCV


# In[84]:

classifier = GridSearchCV(estimator=clf, cv=cv, param_grid=param_grid)


# In[85]:

classifier.fit(X_train, y_train)


# In[86]:

classifier.best_params_, classifier.best_score_


# In[59]:

classifier.score(X_test, y_test)


# In[55]:

from sklearn.model_selection import learning_curve


# In[56]:

title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma


# In[57]:

title


# In[58]:

estimator = svm.SVC(kernel=kernel, gamma=classifier.best_estimator_.gamma)


# In[70]:

classifier.score(X_test, y_test)


# In[71]:

cross_val_score(classifier, X_test, y_test, cv=5)


# In[72]:

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[73]:

plot_learning_curve(estimator, title, X_train, y_train, cv=cv)


# In[74]:

plt.show()


# In[49]:

h=1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[55]:

from sklearn.model_selection import StratifiedKFold


# In[274]:

Z = Z.reshape(xx.shape)


# In[275]:

#plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3,levels=np.linspace(0, 1, xx.shape[0]))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.2)


# In[276]:

plt.xlim(X[:,0].min(), X[:,0].max())
plt.ylim(X[:,1].min(), X[:,1].max())


# In[277]:

plt.show()


# In[30]:

np.logspace(-6, -1, 10), 1/np.logspace(-6, -1, 10)


# In[ ]:




# In[ ]:



