## KiGB : Knowledge Intensive Gradient Boosting

Incorporating richer human inputs including qualitative constraints such as monotonic and synergistic influences has long been adapted inside AI. Inspired by this, we consider the problem of using such influence statements in the successful gradient-boosting framework. We develop a unified framework for both classification and regression settings that can both effectively and efficiently incorporate such constraints to accelerate learning to a better model. Our results in a large number of standard domains and two particularly novel real-world domains demonstrate the superiority of using domain knowledge rather than treating the human as a mere labeler.

KiGB is a unified framework for learning gradient boosted decision trees for regression and classification tasks while leveraging human advice for achieving better performance. Technical details are explained in the [blog](https://starling.utdallas.edu/papers/KiGB). For more details refer the [paper](https://personal.utdallas.edu/~hkokel/pdf/Kokel_AAAI20.pdf) 


This package contains two implementation of Knowledge-intensive Gradient Boosting framework:
- with Gradient Boosted Decision Tree of [Scikit-learn](https://scikit-learn.org) ( SKiGB )
- with Gradient Boosted Decision Tree of [LightGBM](https://github.com/microsoft/LightGBM) ( LKiGB )

Both these implementations are done in python.

## Basic Usage

```python
'''Step 1: Import the class'''
from core.lgbm.lkigb import LKiGB as KiGB

'''Step 2: Import dataset'''
train_data = pd.read_csv('train.csv')
X_train = train_data.drop('target', axis=1)
Y_train = train_data['target']

'''Step 3: Provide monotonic influence information'''
advice  = np.array([1,0,1,1-1], dtype=int)
# 0 for features with no influence, +1 for features with isotonic influence, -1 for antitonic influences

'''Step 4: Train the model'''
kigb = KiGB(lamda=1, epsilon=0.1, advice=advice, objective='regression', trees=30)
kigb.fit(X_train, y_train)

'''Step 5: Test the model'''
kigb.predict(X_test)
```

To use Scikit version of KiGB, import `from core.scikit.skigb import SKiGB`

## Replication

Replication details are available in the experiments section [here](https://github.com/starling-lab/KiGB/blob/master/experiments/README.md)


## Citation

If you build on this code or the ideas of this paper, please use the following citation.

    @inproceedings{kokelaaai20,
      author = {Harsha Kokel and Phillip Odom and Shuo Yang and Sriraam Natarajan},
      title  = {A Unified Framework for Knowledge Intensive Gradient Boosting: Leveraging Human Experts for Noisy Sparse Domains},
      booktitle = {AAAI},
      year   = {2020}
    }


## Acknowledgements

* Harsha Kokel and Sriraam Natarajan acknowledge the support of Turvo Inc. and CwC Program Contract W911NF-15-1-0461 with the US Defense Advanced Research Projects Agency (DARPA)
and the Army Research Office (ARO).

