
#### Replication details


**Standard Baselines**: Comparison of SKiGB performance against SGB (table 2).  


For classification datasets:
```shell script
$python3 ./experiments/classification/compare_vanilla.py
For 'adult' dataset, SKiGB achieved accuracy of '0.855' and SGB achieved accuracy of '0.853'.
For 'australia' dataset, SKiGB achieved accuracy of '0.855' and SGB achieved accuracy of '0.83'.
For 'car' dataset, SKiGB achieved accuracy of '0.984' and SGB achieved accuracy of '0.982'.
For 'cleveland' dataset, SKiGB achieved accuracy of '0.737' and SGB achieved accuracy of '0.677'.
For 'ljubljana' dataset, SKiGB achieved accuracy of '0.696' and SGB achieved accuracy of '0.621'.
```


For regression datasets:
```shell script
$python3 ./experiments/regression/compare_vanilla.py  

For 'abalone' dataset, SKiGB achieved mean-squared error of '5.377' and SGB achieved mean-squared error of '5.491'.
For 'autompg' dataset, SKiGB achieved mean-squared error of '9.793' and SGB achieved mean-squared error of '13.623'.
For 'autoprice' dataset, SKiGB achieved mean-squared error of '8.866' and SGB achieved mean-squared error of '8.945'.
For 'boston' dataset, SKiGB achieved mean-squared error of '24.065' and SGB achieved mean-squared error of '21.493'.
For 'california' dataset, SKiGB achieved mean-squared error of '47.159' and SGB achieved mean-squared error of '47.468'.
For 'cpu' dataset, SKiGB achieved mean-squared error of '0.185' and SGB achieved mean-squared error of '0.204'.
For 'crime' dataset, SKiGB achieved mean-squared error of '2.211' and SGB achieved mean-squared error of '2.296'.
For 'redwine' dataset, SKiGB achieved mean-squared error of '0.381' and SGB achieved mean-squared error of '0.419'.
For 'whitewine' dataset, SKiGB achieved mean-squared error of '0.426' and SGB achieved mean-squared error of '0.439'.
For 'windsor' dataset, SKiGB achieved mean-squared error of '3.9' and SGB achieved mean-squared error of '4.626'.
```
 

**Monotonic Baselines**:  Comparison of KiGB against monotonic boosting approaches (table 3 & 4).
 
 
 SKiGB comparision for classification datasets with Monoensemble (MONO):
 ```shell
$python3 ./experiments/classification/compare_monoensemble.py
For 'adult' dataset, SKiGB achieved accuracy of '0.855' and Monoensemble achieved accuracy of '0.857'.
For 'australia' dataset, SKiGB achieved accuracy of '0.855' and Monoensemble achieved accuracy of '0.884'.
For 'car' dataset, SKiGB achieved accuracy of '0.984' and Monoensemble achieved accuracy of '0.765'.
For 'cleveland' dataset, SKiGB achieved accuracy of '0.737' and Monoensemble achieved accuracy of '0.74'.
For 'ljubljana' dataset, SKiGB achieved accuracy of '0.696' and Monoensemble achieved accuracy of '0.611'.
``` 

LKiGB comparision for classification datasets with LMC:
 ```shell
$python3 ./experiments/classification/compare_lmc.py
.
[LIGHTGBM] [Warning] ...
.
For 'adult' dataset, LKiGB achieved accuracy of '0.865' and LMC achieved accuracy of '0.863'.
For 'australia' dataset, LKiGB achieved accuracy of '0.878' and LMC achieved accuracy of '0.867'.
For 'car' dataset, LKiGB achieved accuracy of '0.971' and LMC achieved accuracy of '0.959'.
For 'cleveland' dataset, LKiGB achieved accuracy of '0.757' and LMC achieved accuracy of '0.73'.
For 'ljubljana' dataset, LKiGB achieved accuracy of '0.721' and LMC achieved accuracy of '0.718'.
``` 
 
 
LKiGB comparision for regression datasets with LMC:
 ```shell
$python3 ./experiments/regression/compare_lmc.py
.
[LIGHTGBM] [Warning] ...
.
For 'abalone' dataset, LKiGB achieved mean-squared error of '4.786' and LMC achieved mean-squared error of '4.797'.
For 'autompg' dataset, LKiGB achieved mean-squared error of '8.047' and LMC achieved mean-squared error of '8.33'.
For 'autoprice' dataset, LKiGB achieved mean-squared error of '14.953' and LMC achieved mean-squared error of '15.614'.
For 'boston' dataset, LKiGB achieved mean-squared error of '15.496' and LMC achieved mean-squared error of '16.292'.
For 'california' dataset, LKiGB achieved mean-squared error of '48.517' and LMC achieved mean-squared error of '50.94'.
For 'cpu' dataset, LKiGB achieved mean-squared error of '0.206' and LMC achieved mean-squared error of '0.208'.
For 'crime' dataset, LKiGB achieved mean-squared error of '1.834' and LMC achieved mean-squared error of '1.847'.
For 'redwine' dataset, LKiGB achieved mean-squared error of '0.382' and LMC achieved mean-squared error of '0.397'.
For 'whitewine' dataset, LKiGB achieved mean-squared error of '0.45' and LMC achieved mean-squared error of '0.467'.
For 'windsor' dataset, LKiGB achieved mean-squared error of '2.524' and LMC achieved mean-squared error of '2.634'.
``` 

