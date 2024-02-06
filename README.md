#### Overview
LDA and Principal Component Analysis (PCA) are two techniques
for dimensionality reduction. PCA can be decribed as an unsupervised algorithm that ignores data labels and aims to find directions which maximalize
the variance in a data. In comparison with PCA, LDA is a supervised algorithm and aims to project a dataset onto a lower dimensional space with good
class separability. In other words, LDA maximalizes the ratio of betweenclass variance and the within-class variance in a given data.

#### Linear Discriminant Analysis
LDA finds directions where classes are well-separated,
i.e. LDA maximizes the ratio of between-class variance and the within-class
variance. Firstly, assume that $C$ is a set of classes and set $D$, which represents
a training dataset, is defined as $D = \{x_1, x_2, . . . , x_N \}$.

The between-classes scatter matrix SB is defined as:
$S_b = \sum_c N_C(\mu_c -\overline{x})(\mu_c - \overline{x})^T$, where $\overline{x}$ is a vector represents the overall mean of the data, µ represents the mean corresponding to each class, and $N_C$ are sizes of the respective classes.

The within-classes scatter matrix $S_W$ is defined as:

$S_W = \sum_c \sum_{x \in D_c}(x - \overline{\mu_c})(x - \overline{\mu_c})^T$

Next, we will solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$ to obtain the linear discriminants, i.e.

$(S_W^{-1}S_B)w = \lambda w$

where $w$ represents an eigenvector and $\lambda$ represents an eigenvalue. Finally,
choose k eigenvectors with the largest eigenvalue and transform the samples
onto the new subspace.

##### Compute the within-scatter matrix 
#number of classes = n
$S_W = \sum_c \sum_{x \in D_c}(x - \overline{\mu_c})(x - \overline{\mu_c})^T$

##### Compute the between-scatter matrix
$S_b = \sum_c N_C(\mu_c -\overline{x})(\mu_c - \overline{x})^T$

##### Solve the EigenProblem and return eigen-vector
```{r}
SolveEigenProblem <- function(withinMatrix, betweenMatrix, prior)
{
  # Sw^-1 * Sb solve: https://www.geeksforgeeks.org/inverse-of-matrix-in-r/?ref=lbp, https://www.geeksforgeeks.org/solve-linear-algebraic-equation-in-r-programming-solve-function/
  eivectors = eigen(solve(withinMatrix) %*% betweenMatrix)
  return(eivectors)
}
```

##### Visualize the results
Data are projected into lower-dimensional subspace. TODO

##### Results
The ComputeWithinScatter and ComputeBetweenScatter functions were modified to include the label parameter, and the ComputeBetweenScatter function was also modified to include the mean parameter, because without these modifications, errors were continuously thrown.

We experimented with two techniques for dimensionality reduction, PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis), using a provided dataset of wines. Each wine in the dataset is characterized by the following thirteen attributes:

- Alcohol
- Malic Acid
- Ash
- Alkalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of Diluted Wines
- Proline

After analyzing the mentioned dataset (e.g., by outputting values), we can observe significant differences between the values of attributes V2-V14, where higher values are noted for attributes like V14 and V11.
The training accuracy value matches the classification accuracy mentioned in the wine_info.txt document. However, since the accLDA value is not equal to 1, it indicates that the wines are not perfectly linearly separated into classes. This means that it would be appropriate to consider a more suitable method for the given wine dataset.
According to: 
"The data was used with many others for comparing various classifiers. The classes are separable, though only RDA has achieved 100% correct classification. (RDA: 100%, QDA: 99.4%, LDA: 98.9%, 1NN: 96.1% (z-transformed data)) (All results using the leave-one-out technique)"
A more suitable method might be, for example, QDA with a 99.4% accuracy or RDA.
