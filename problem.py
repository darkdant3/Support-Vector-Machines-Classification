import numpy as np
import matplotlib
# matplotlib.use('MACOSX')
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report


class SVMClassification:
    pass
dataset = np.genfromtxt('input3.csv', delimiter=',', skip_header=1)
X = dataset[:, 0:2]
y = dataset[:, -1]

fig, ax = plt.subplots()
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAcAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='+', c=y[y == 0], cmap=cmap_light)
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c=y[y == 1], cmap=cmap_bold)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)


kernel = 'linear'
gammas = np.logspace(-6, -1, 10)
clf = svm.SVC()
C = [0.1, 0.5, 1, 5, 10, 50, 100]
cv = 5
tuned_parameters = [
  #{'C': C, 'kernel': ['linear']},
  {'C': C, 'gamma':  [0.1, 0.5, 1, 3, 6, 10], 'kernel': ['rbf']},
 #{'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma':[0.1, 1], 'kernel': ['poly']},
 ]


cv = 5
clf = GridSearchCV(estimator=svm.SVC(C=1), cv=cv, param_grid=tuned_parameters)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' % clf.best_estimator_.gamma
estimator = svm.SVC(kernel=clf.best_estimator_.kernel, C=clf.best_estimator_.C, gamma=clf.best_estimator_.gamma)
print("Cross validation scores on test set")
print cross_val_score(estimator, X_test, y_test)


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
        :class:`StratifiedKFold` used. If the estimator is not a clf
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


h = 1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cmap_bold, alpha=0.1)

plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
plt.show()
