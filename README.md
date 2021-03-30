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
import pandas as pd
import numpy as np

'''Step 2: Import dataset'''
train_data = pd.read_csv('datasets/classification/car/train_0.csv')
X_train = train_data.drop('class', axis=1)
Y_train = train_data['class']
test_data = pd.read_csv('datasets/classification/car/test.csv')
X_test = train_data.drop('class', axis=1)
Y_test = train_data['class']


'''Step 3: Provide monotonic influence information'''
advice  = np.array([-1,-1,0,+1,0,+1], dtype=int)
# 0 for features with no influence, +1 for features with isotonic influence, -1 for antitonic influences

'''Step 4: Train the model'''
kigb = KiGB(lamda=1, epsilon=0.1, advice=advice, objective='binary', trees=30)
kigb.fit(X_train, Y_train)

'''Step 5: Test the model'''
Y_pred = kigb.predict(X_test)
feature_importance = kigb.feature_importance()

```

To use Scikit version of KiGB, import `from core.scikit.skigb import SKiGB`

To perform regression, use `objective='regression'`.

## Rendering Tree

LKiGB trees can be rendered using following commands:

```python
import lightgbm as lgb
tree_0 = lgb.create_tree_digraph(kigb.kigb, tree_index=0, name='tree_0')
tree_0.render("tree_0")
```

Or 


```python
import lightgbm as lgb
import matplotlib.pyplot as plt
ax = lgb.plot_tree(kigb.kigb, tree_index=0, figsize=(20, 20))
plt.show()
```

SKiGB trees can be rendered to a file using following commands:

```python  
from core.scikit.utils import export_tree
feature_names = X_train.columns.values
export_tree(kigb=kigb, tree_index=0, feature_names=feature_names, filename="tree_0.png")
```

**Note**: `tree_index` takes the index of the tree to be rendered.

## Replication

Replication details are available in the experiments section [here](https://github.com/starling-lab/KiGB/blob/master/experiments/README.md)

## Steps to debug

To debug the KiGB, or analyze how the expected value of leaf node is changed by the penalty, configure the log level to `INFO` before training the model (Step 4). 

```python
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)
```   

This should provide additional information in logs as shown below, for each tree.  

```
old leaves: leaf_value=0.12, 0.23, 0.32, 0.9
new leaves: leaf_value=-0.1, 0.02, 0.3, 0.7
```

Log level `DEBUG` provides further details about which constraint (isotonic or antitonic) was violated at which node in the tree. It also provides the penalty term, violation and sample size for each left and right child of the violated node as shown below. To see all that information, use `level=logging.DEBUG` in the above logging configuration.

```
antitonic constraint not satisfied for tree 0 node 2
left penalty: 0.0155 sample: 20000 violation: 0.062
right penalty: 0.002 sample: 30000 violation: 0.0122
```

**Note:** penalty = (lamda * violation)/(sample * 2)   




## Citation

If you build on this code or the ideas of this paper, please use the following citation.

    @inproceedings{kokelaaai20,
     	title={A Unified Framework for Knowledge Intensive Gradient Boosting: Leveraging Human Experts for Noisy Sparse Domains}, 
	    volume={34}, 
    	DOI={10.1609/aaai.v34i04.5873}, 
	    number={04}, 
    	journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	    author={Kokel, Harsha and Odom, Phillip and Yang, Shuo and Natarajan, Sriraam}, 
    	year={2020}, 
		pages={4460-4468} 
    }


## Acknowledgements

* Harsha Kokel and Sriraam Natarajan acknowledge the support of Turvo Inc. and CwC Program Contract W911NF-15-1-0461 with the US Defense Advanced Research Projects Agency (DARPA)
and the Army Research Office (ARO).

