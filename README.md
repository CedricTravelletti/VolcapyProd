# Volcapy: Bayesian Inversion for large scale Geophysical Inverse Problems (and beyond ...)


The Bayesian approach to inverse problems is a well known and powerful one [[1]](#1), but it often fails to scale due to the fact that one has to handle covariance matrices of size **n^2**, where **n** is the size of the model discretization.

For example, it is well known that Gaussian Process (GP) based inversion techniques have a complexity of *O(n^3)* and thus gets intractable at moderate grid sizes.


The aim of the Volcapy project is to provide GP-based Bayesian inversion algorithms that scale to grids with over *100k* cells.


For more information, please see the [Project Website](https://cedrictravelletti.github.io/Volcano/)


*The name Volcapy comes from the origin of the project as an inversion code for volcano gravimetry. It has now evolved into a general large scale Bayesian inversion framework.*


## References
<a id="1">[1]</a> 
Tarantola, A. (2005). 
Inverse problem theory and methods for model parameter estimation. 
SIAM, volume 89.

