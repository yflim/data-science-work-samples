This repository contains mostly data science-related assignment solutions, in the form of Python code, Jupyter notebooks, and PDFs rendered from Jupyter markdown for derivations/proofs, as well as datasets as applicable. Nothing here was produced following any relevant solutions that may be available on the web, with the current exception of a few lines of utility code.

Note that some modules implemented in this repository have been edited, and code in some of the notebooks has been moved into modules, but affected/dependent notebooks have sometimes not yet been updated accordingly. In other words, code in this repository may not currently be in a consistent state.

Not included due to NDA: TransLink unsupervised clustering notebooks.

## Table of folder contents

### Models and algorithms

[Clustering](https://github.com/yflim/data-science-work-samples/tree/master/clustering)
- Centroid-based clustering: Notebook implementing and using k-means, k-means++, k-medians, and k-medoids/PAM from scratch.

[Cross-validation](https://github.com/yflim/data-science-work-samples/tree/master/cross_validation)
- Leave-one-out and k-fold cross-validation methods for classification, efficiently recomputing moments (expectation and variance) in each iteration. Regression not currently supported.

[Decision tree](https://github.com/yflim/data-science-work-samples/tree/master/decision_tree)
- Basic implementation from scratch with support for both regression and classification, including cost complexity pruning.

[Linear models: logistic regression implementation](https://github.com/yflim/data-science-work-samples/tree/master/linear_model)
- Logistic regression class from scratch, with L2 regularisation support.

[Logistic regression](https://github.com/yflim/data-science-work-samples/tree/master/logistic_regression)
- Parameter MLE from scratch for logistic regression using the Newton-Raphson method, plus out-of-the-box Scikit-learn solution.

[Neural networks](https://github.com/yflim/data-science-work-samples/tree/master/neural_networks)
- [Implementation](https://github.com/yflim/data-science-work-samples/tree/master/neural_networks/implementation): Implementation of simple dense network class from scratch, with demo notebook.
- [Simple Keras usage](https://github.com/yflim/data-science-work-samples/tree/master/neural_networks/simple_keras): Building and tuning a dense network to solve a binary classification problem.
- [Sentiment analysis](https://github.com/yflim/data-science-work-samples/tree/master/neural_networks/sentiment_analysis_WIP): Sentiment analysis of the [Sentiment140](http://help.sentiment140.com/for-students) dataset of 1.6 million Tweets.

[OLS and GLM](https://github.com/yflim/data-science-work-samples/tree/master/OLS_and_GLM)
- OLS solution from scratch using gradient descent, with comparison against ready-made solution from statsmodels.
- Some related theory; also for GLM.

[Optimisation](https://github.com/yflim/data-science-work-samples/tree/master/optimise)
- Currently contains basic gradient descent implementation

[Random forest](https://github.com/yflim/data-science-work-samples/tree/master/random_forest)
- Basic implementation from scratch with support for both regression and classification. NOT REALLY TESTED YET.

[Regularisation](https://github.com/yflim/data-science-work-samples/tree/master/regularisation)
- Notebook demonstrating L2 regularisation in logistic regression (implemented in [linear_model/logistic_regression.py](https://github.com/yflim/data-science-work-samples/blob/master/linear_model/logistic_regression.py)).

### Other

[Data](https://github.com/yflim/data-science-work-samples/tree/master/data)
- Datasets used

[Utils](https://github.com/yflim/data-science-work-samples/blob/master/utils/utils.py)
- Generally useful support methods
