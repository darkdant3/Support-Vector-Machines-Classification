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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class SVMClassification:

    def __init__(self):
        self.param_grid = [{'kernel': 'linear', 'C': 1}]
        self.cv = 5
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.clf = None
        self.estimator = svm.SVC(C=1)
        self.method = 'linear'


    def run_gridsearch(self):

        if self.method == 'knn':
            #print self.estimator.get_params().keys()
            #print self.param_grid
            #clf = GridSearchCV(self.estimator, cv=self.cv, param_grid=[self.param_grid])
            clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5)
        else:
            clf = GridSearchCV(estimator=self.estimator, cv=self.cv, param_grid=[self.param_grid])

        clf.fit(self.X_train, self.y_train)
        print("Best parameters set found on development set:")
        print ''
        print(clf.best_params_)
        print ''
        print("Grid scores on development set:")
        print ''
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print ''

        print("Detailed classification report:")
        print ''
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print ''
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print(classification_report(y_true, y_pred))
        print ''
        self.clf = clf


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
    plt.subplot(211)
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


def plot_decision_boundary(X, y, clf, plt):

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAcAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.subplot(212).scatter(X[y == 0, 0], X[y == 0, 1], marker='+', c=y[y == 0], cmap=cmap_light, label='A')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('Decision boundary plot')
    plt.subplot(212).scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c=y[y == 1], cmap=cmap_bold, label='B')
    h = 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(212).contourf(xx, yy, Z, cmap=cmap_bold, alpha=0.1)

# Load and split dataset
dataset = np.genfromtxt('input3.csv', delimiter=',', skip_header=1)
X = dataset[:, 0:2]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
C = [0.1, 0.5, 1, 5, 10, 50, 100]
svc = svm.SVC(C=1)
log_regression = LogisticRegression(C=[0.1, 0.5, 1, 5, 10, 50, 100])

tuned_parameters = [{'n_neighbors': range(1, 51, 1), 'leaf_size': range(5, 65, 5),
'weights': ['distance', 'uniform'],
#'weights': ['distance'],
'algorithm': ['ball_tree', 'brute']}]
#knn = KNeighborsClassifier(n_neighbors=range(5, 65, 5), leaf_size=range(5, 65, 5))
knn = KNeighborsClassifier(n_jobs=2)
dt_params = {'max_depth': range(1, 51, 1), 'min_samples_split': range(2, 11)}
decision_tree = DecisionTreeClassifier(max_depth=range(1, 51, 1), min_samples_split=range(2, 11))
random_forest = RandomForestClassifier(max_depth=range(1, 51, 1), min_samples_split=range(2, 11))
method_params = [
        {'method': 'svm_linear', 'estimator': svc, 'params': {'C': C, 'kernel': ['linear']}},
        {'method': 'svm_rbf', 'estimator': svc, 'params': {'C': C, 'gamma':  [0.1, 0.5, 1, 3, 6, 10], 'kernel': ['rbf']}},
        {'method': 'svm_poly', 'estimator': svc, 'params': {'C': [0.1, 1, 3], 'degree':[4, 5, 6], 'gamma':[0.1, 1], 'kernel': ['poly']}},
        {'method': 'logistic', 'estimator': log_regression, 'params': {'C': C}},
        {'method': 'knn', 'estimator': knn, 'params': {'n_neighbors': np.arange(1, 51, 1)}},
        {'method': 'decision_tree', 'estimator': decision_tree, 'params': dt_params},
        {'method': 'random_forest', 'estimator': random_forest, 'params': dt_params},
    ]

output = open('output3.csv', 'w')
for mp in method_params:

    svm_clf = SVMClassification()
    svm_clf.X = X
    svm_clf.y = y
    svm_clf.X_train = X_train
    svm_clf.y_train = y_train
    svm_clf.X_test = X_test
    svm_clf.y_test = y_test
    svm_clf.estimator = mp.get('estimator')
    svm_clf.param_grid = mp.get('params')
    svm_clf.method = mp.get('method')

    print '\nRunning : %s algo......\n' % mp.get('method')
    start_time = time.time()
    svm_clf.run_gridsearch()
    elapsed = time.time() - start_time
    print '**%s gridsearch took %.8fs ' % (mp.get('method'), elapsed)

    title = 'Learning Curve  %s {}'.format(svm_clf.clf.best_params_) % (mp.get('method'))
    if mp.get('method') is 'svm_linear':
        estimator = svm.SVC(kernel=svm_clf.clf.best_estimator_.kernel, C=svm_clf.clf.best_estimator_.C, gamma=svm_clf.clf.best_estimator_.gamma)

    elif mp.get('method') is 'logistic':
        estimator = svm.SVC(C=svm_clf.clf.best_estimator_.C, probability=True)
    elif mp.get('method') is 'knn':
        estimator = KNeighborsClassifier(
            n_neighbors=svm_clf.clf.best_params_.get('n_neighbors'),
            algorithm=svm_clf.clf.best_params_.get('algorithm'),
            weights=svm_clf.clf.best_params_.get('weights'),
            leaf_size=svm_clf.clf.best_params_.get('leaf_size'))
    elif mp.get('method') is 'decision_tree':
        estimator = DecisionTreeClassifier(max_depth=svm_clf.clf.best_params_.get('max_depth'),
            min_samples_split=svm_clf.clf.best_params_.get('min_samples_split'))
    elif mp.get('method') is 'random_forest':
        estimator = RandomForestClassifier(max_depth=svm_clf.clf.best_params_.get('max_depth'),
            min_samples_split=svm_clf.clf.best_params_.get('min_samples_split'))
    else:
        estimator = svm.SVC(kernel=svm_clf.clf.best_estimator_.kernel, C=svm_clf.clf.best_estimator_.C, gamma=svm_clf.clf.best_estimator_.gamma)
    print("Cross validation avg. score on test set")
    scores = cross_val_score(estimator, X_test, y_test)
    print scores.mean()
    estimator.fit(X_train, y_train)
    #plt = plot_learning_curve(estimator, title, X_train, y_train, cv=5)
    #plot_decision_boundary(X, y, estimator, plt)

    line = '%s,%.3f,%.3f' % (mp.get('method'), svm_clf.clf.best_score_, scores.mean())
    output.write(line + "\n")
    output.flush()
output.close()
#plt.show()
