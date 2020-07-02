# Pattern regression implementations

The codes for pattern regression based individualized behavioral prediction analysis. I have made substantial changes based on our previous codes in paper ```Cui and Gong, 2018, NeuroImage```. I have tested these codes. If there is any problems, send email to zaixucui@gmail.com.

We included detailed documentation of ridge regression and relevance vector regression (RVR) in the wiki https://github.com/ZaixuCui/Pattern_Regression_Clean/wiki.
The usage of codes of linear regression, lasso and elastic-net is similar to that of ridge regression (https://github.com/ZaixuCui/Pattern_Regression_Clean/wiki/Ridge-regression-manual).
The usage of codes of support vector regression is similar to that of relevance vector regression (https://github.com/ZaixuCui/Pattern_Regression_Clean/wiki/RVR-manual).

Citing our related paper will be greatly appreciated if you use these codes.
<br>&emsp; ```Zaixu Cui, Gaolang Gong, The effect of machine learning regression algorithms and sample size on individualized behavioral prediction with functional connectivity features, (2018), NeuroImage, 178: 622-37```
<br>&emsp; ```Zaixu Cui, et al., Individualized Prediction of Reading Comprehension Ability Using Gray Matter Volume, (2018), Cerebral Cortex, 28(5):1656–72```
<br>&emsp; ```Zaixu Cui, et al., Individual variation in functional topography of association networks in youth. (2020) Neuron, 106(2): 340-53.```
<br>&emsp; ```Zaixu Cui, et al., Optimization of energy state transition trajectory supports the development of executive function during youth. (2020) eLife. 9:e53060. ```

The scikit-learn library (version: 0.16.1) was used to implement OLS regression, LASSO regression, ridge regression and elastic-net regression (http://scikit-learn.org/) (Pedregosa et al., 2011), the LIBSVM function in MATLAB was used to implement LSVR (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Chang and Lin, 2011), the PRoNTo toolbox (http://www.mlnl.cs.ucl.ac.uk/pronto/) was used to implement RVR (Schrouff et al., 2013). 
Note: Codes for RVR will not work well in Matlab higher than 2012 version.

Generally, I like ridge regression and relevance vector regression algorithms, see ```Cui and Gong, 2018, NeuroImage```.
