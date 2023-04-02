# Common Machine Learning (ML) Algorithms

Concise catalog of ML algorithms and preprocessing techniques. This is intended to be a one-stop location for the initial review of documentation to synthesize some of the online information related to proven state-of-the-art research.

Broadly, there are 3 types of Machine Learning Algorithms: supervised, unsupervised, and reinforcement learning. Which one to choose will greatly depend in the data structure and the features of interest. In the context of machine learning, discrete variables are used for classification, and continuous variables are used for regression. This is an excellent reference for satellite imagery machine learning applications, [satellite-image-deep-learning](https://github.com/robmarkcole/satellite-image-deep-learning).

## List of Common Machine Learning Algorithms

**1. Linear Regression**

- Here, we establish relationship between independent and dependent variables by fitting a best line. This best fit line is known as regression line and represented by a linear equation Y= a *X + b. It is used to estimate real values (NDVI, tree height, etc.) based on continuous variable(s). Simple Linear Regression and Multiple Linear Regression are two types of regression algorithms.
- Article: Mark, David M., and Michael Church. "On the misuse of regression in earth science." Journal of the International Association for mathematical geology 9.1 (1977): 63-75.
- [Code Example](https://github.com/rapidsai/cuml/blob/branch-22.02/notebooks/linear_regression_demo.ipynb)

**2. Logistic (Logit) Regression**

- It is a classification not a regression algorithm. It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function. Hence, it is also known as logit regression. Since, it predicts the probability, its output values lies between 0 and 1 (as expected).
- Article: Wang, Liang-Jie, et al. "Landslide susceptibility mapping in Mizunami City, Japan: A comparison between logistic regression, bivariate statistical analysis and multivariate adaptive regression spline models." Catena 135 (2015): 271-282.
- [Code Example](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Logistic%20Regression%20in%20Python%20-%20Step%20by%20Step.ipynb)

**3. Decision Tree**

- It is a type of supervised learning algorithm that is mostly used for classification and regression problems. It works for both categorical and continuous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/ independent variables to make as distinct groups as possible. MARS is a type of single decission trees. Algorithms such as Bagging or Random Forest create ensembles of decision trees making them more robust.
- Article: Li, Ting, et al. "Landslide susceptibility mapping using random forest." Geography and Geo-Information Science 30.6 (2014): 1250-1256.
- [Code Example](https://github.com/rapidsai/cuml/blob/branch-22.02/notebooks/random_forest_demo.ipynb)

**4. Support Vector Machine (SVM)**

- It is a classification method. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we will find some line that splits the data between the two differently classified groups of data. This will be the line such that the distances from the closest point in each of the two groups will be farthest away. Then, depending on where the testing data lands on either side of the line, that’s what class we can classify the new data as.
- Article: Abdollahi, Abolfazl, Hamid Reza Riyahi Bakhtiari, and Mojgan Pashaei Nejad. "Investigation of SVM and level set interactive methods for road extraction from google earth images." Journal of the Indian Society of Remote Sensing 46.3 (2018): 423-430.
- [Code Example](https://medium.com/rapids-ai/fast-support-vector-classification-with-rapids-cuml-6e49f4a7d89e)

**5. Naive Bayes**

- It is a classification technique based on Bayes’ theorem with an assumption of independence between predictors. A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability of the feature in question. Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods. This algorithm is mostly used in text classification and with problems having multiple classes.
- Article: Abu El-Magd, Sherif Ahmed, Sk Ajim Ali, and Quoc Bao Pham. "Spatial modeling and susceptibility zonation of landslides using random forest, naïve bayes and K-nearest neighbor in a complicated terrain." Earth Science Informatics 14.3 (2021): 1227-1243.
- [Code Example](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb)

**6. K-Nearest Neighbors (kNN)**

- Is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. The KNN algorithm assumes that similar things exist in close proximity. To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.
- Article: Lee, Taylor R., Warren T. Wood, and Benjamin J. Phrampus. "A machine learning (kNN) approach to predicting global seafloor total organic carbon." Global Biogeochemical Cycles 33.1 (2019): 37-46.
- [Code Example](https://medium.com/rapids-ai/accelerating-k-nearest-neighbors-600x-using-rapids-cuml-82725d56401e)

**7. K-Means**

- K-Means is a clustering algorithm. Is an iterative algorithm that tries to partition the dataset into K pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. The approach kmeans follows to solve the problem is called Expectation-Maximization. The E-step is assigning the data points to the closest cluster. Data points inside a cluster are homogeneous and heterogeneous to peer groups.
- Article: Hua, Chen, et al. "Application of K-means classification in remote sensing [J]." Infrared and Laser Engineering 2 (2000).
- [Code Example](https://github.com/rapidsai/cuml/blob/branch-22.02/notebooks/kmeans_demo.ipynb)

**8. Dimensionality Reduction Algorithms**

- Dimensionality reduction is an unsupervised learning technique. It refers to techniques for reducing the number of input variables in training data, normally used as a data preparation technique performed on data prior to modeling. It might be performed after data cleaning and data scaling and before training a predictive model. Some algorithms for dimensionality reduction include Linear Algebra and Manifold Learning methods.
    - Linear Algebra: Principal Components Analysis, Singular Value Decomposition, Non-Negative Matrix Factorization
    - Manifold Learning: Isomap Embedding, Locally Linear Embedding, Multidimensional Scaling, Spectral Embedding, t-distributed Stochastic Neighbor Embedding
- Article: Khodr, Jihan, and Rafic Younes. "Dimensionality reduction on hyperspectral images: A comparative review based on artificial datas." 2011 4th International Congress on Image and Signal Processing. Vol. 4. IEEE, 2011.
- [Code Example](https://github.com/junjun-jiang/SuperPCA)

**9. Gradient Boosting algorithms**

- Gradient boosting can be used for both classification and regression problems. It involves three elements: a loss function to be optimized, a weak learner to make predictions, and an additive model to add weak learners to minimize the loss function. It can benefit from regularization methods that penalize various parts of the algorithm and generally improve the performance of the algorithm by reducing overfitting. Some of the most common algorithms for gradient boosting are: GBM, XGBoost, LightGBM, CatBoost
- Article: Li, Yinxing, and Minghao Liu. "Spatialization of population based on Xgboost with multi-source data." IOP Conference Series: Earth and Environmental Science. Vol. 783. No. 1. IOP Publishing, 2021.
- [Code Example](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html)

**10. Neural Networks**

- Neural networks can be used for either regression or classification. They are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network. 
- Article: Biswajeet, Pradhan, and Lee Saro. "Utilization of optical remote sensing data and GIS tools for regional landslide hazard analysis using an artificial neural network model." Earth Science Frontiers 14.6 (2007): 143-151.
- [Code Example](https://github.com/pytorch/examples/tree/master/imagenet)

**11. Long Short Term Memory (LSTM)**

- A type of Recurrent Neural Networks (RNN) capable of learning long-term dependencies used in both regression and classification. It processes data passing on information as it propagates forward, by being structured in three parts. The first part chooses whether the information coming from the previous timestamp is to be remembered or is irrelevant and can be forgotten. In the second part, the cell tries to learn new information from the input to this cell. At last, in the third part, the cell passes the updated information from the current timestamp to the next timestamp.
- Article: Mohan, Arvind, et al. "Compressed convolutional LSTM: An efficient deep learning framework to model high fidelity 3D turbulence." arXiv preprint arXiv:1903.00033 (2019).
- [Code Example](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/)

**12. Autoencoders**

- Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning. We can take an unlabeled dataset and frame it as a supervised learning problem tasked with outputting y, a reconstruction of the original input x. This network can be trained by minimizing the reconstruction error, which measures the differences between our original input and the consequent reconstruction. Because neural networks are capable of learning nonlinear relationships, this can be thought of as a more powerful (nonlinear) generalization of PCA. Some use cases are anomaly detection, data denoising (ex. images, audio), image inpainting, information retrieval, and others.
- Article: Li, Lianfa. "Deep residual autoencoder with multiscaling for semantic segmentation of land-use images." Remote Sensing 11.18 (2019): 2142.
- [Code Example](https://towardsdatascience.com/autoencoders-for-land-cover-classification-of-hyperspectral-images-part-1-c3c847ebc69b)

**13. Convolutional Neural Networks (CNNs)**

- CNNs are normally used for classification problems, but with some work can be used for regression problems as well. CNNs can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
- Article: Maskey, Manil, Rahul Ramachandran, and Jeffrey Miller. "Deep learning for phenomena-based classification of Earth science images." Journal of Applied Remote Sensing 11.4 (2017): 042608.
- [Code Example](https://towardsdatascience.com/land-cover-classification-of-satellite-imagery-using-convolutional-neural-networks-91b5bb7fe808)

**14. Generative Adversarial Networks (GANs)**

- Generative Adversarial Networks belong to the set of generative models. It means that they are able to produce / to generate new content. GANs are algorithmic architectures that use two neural networks, one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. They are used widely in image generation, video generation and voice generation.
- Article: Li, Yuanhao, et al. "Super-resolution of geosynchronous synthetic aperture radar images using dialectical GANs." Science China Information Sciences 62.10 (2019): 1-3.
- [Code Example](https://github.com/eriklindernoren/PyTorch-GAN)

**15. Transformers**

- Transformers can be used for sequence problems, as well as segmentation. Like LSTM, Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the previously described/existing sequence-to-sequence models because it does not imply any Recurrent Networks (GRU, LSTM, etc.). Both Encoder and Decoder are composed of modules that can be stacked on top of each other multiple times, which is described by Nx in the figure.
- Article: Hong, Danfeng, et al. "Spectralformer: Rethinking hyperspectral image classification with transformers." IEEE Transactions on Geoscience and Remote Sensing (2021).
- [Code Example](https://github.com/open-mmlab/mmsegmentation)
