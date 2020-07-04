# Stock_Investment

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/dollar-517113_960_720.webp)

# (I) Abstract : 

Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stock's future price could yield significant profit. The efficient market hypothesis posits that stock prices are a function of information and rational expectations, and that newly revealed information about a company's prospects is almost immediately reflected in the current stock price. Predicting how the stock market will perform is one of the most difficult things to do. There are so many factors involved in the prediction – physical factors vs. physhological, rational and irrational behaviour, etc. All these aspects combine to make share prices volatile and very difficult to predict with a high degree of accuracy. 

In this endeavor we will work with historical data about the stock prices of few publicly listed companies and implement machine learning based on **Long Short Term Memory(LSTM)** to predict the future stock prices. 

# (II) Dataset Used : 

The dataset was made publically available on *Kaggle* platform. The Google's stock prices were taken from *SuperDataScience* webpage. 
* **AMD** : **https://www.kaggle.com/gunhee/amdgoogle**
* **Tesla** : **https://www.kaggle.com/rpaguirre/tesla-stock-price**

The others have been uploaded in their respective subfolder for easy access. 

Some stock insights from the datasets : 

**AMD** :
![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/AMD.png)

**Tesla** : 
![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/TESLA.png)

**Amazon** : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/Amazon.png)

# (III) Libraries Used : 

## Numpy : 

Fundamental package for scientific computing in Python3, helping us in creating and managing n-dimensional tensors. A vector can be regarded as a 1-D tensor, matrix as 2-D, and so on. 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/tensor.jpg) 

## Pandas : 

Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/pandas.jpeg)

## Matplotlib : 

Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+. 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/matplotlib_intro.png)

## Seaborn : 

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. 

## Tensorflow's Keras API

Is an open source deep learning framework for dataflow and differentiable programming. It’s created and maintained by Google.

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/tfkeras.png)

## TQDM : 

Is a progress bar library with good support for nested loops and Jupyter notebooks.

## Sci-kit learn : 

Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines and many more.

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/boston-dataset-scikit-learn-machine-learning-python-tutorial.png) 

# (IV) Exploratory Data Analysis(EDA) : 

When we’re getting started with a machine learning (ML) project, one critical principle to keep in mind is that data is everything. It is often said that if ML is the rocket engine, then the fuel is the (high-quality) data fed to ML algorithms. However, deriving truth and insight from a pile of data can be a complicated and error-prone job. To have a solid start for our ML project, it always helps to analyze the data up front.

During EDA, it’s important that we get a deep understanding of:

* The **properties of the data**, such as schema and statistical properties;
* The **quality of the data**, like missing values and inconsistent data types;
* The **predictive power of the data**, such as correlation of features against target.

This project didn't require profound EDA as the data was *time-series*. Only thing to enusure in every dataset from AMD to Tesla was to ensure we don't have missing values. Fortunately, that didn't turn out to be true.

Using pandas *info()* function of the dataframe structure we found that all rows of the **Open** prices were filled. 

Excerpt from AMD's data :

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/AMD%20entries.png)

Excerpt from Tesla's data :

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/Tesla%20entries.png)

Also, open, close, day's max and day's min were found highly correlated which is otherwise obvious too. However, just to visualize this we plotted the correlation extent using Seaborn's **heatmap**.

*Tesla's heatmap*

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/tesla_heapmap.png)

*AMD's heaptmap*

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/AMD_heatmap.png)

This heat map could be used in order to understand the *available stock volume's* correlation with other prices(open, close, max, min) for future applications. However, in this project, we keep ourselves to *open stock prices* prediction based on historical data. 

# (V) Hyperparameters :

Our machine learning model is based on two hyperparameters which are :
* **Time Step** : Number of days in the past our model will look at in order to predict the price on the present day. For illustration, if we set *time_step = 7* then for predicting the price on **n th** day, our model will analyze all the prices from **n-1** to **n-7** days. This approach is relatively more accurate than using a traditional machine learning algorithm - *such as polynomial linear regression* - as we consider only recently reported prices rather than the whole dataset at once. 
* **Days** : Number of days in the end for which we have to predict the prices for. These will be in our validation/test set.

# (VI) Scaling : 

It refers to putting the values in the same range or same scale so that no variable is dominated by the other.

Most of the times, our dataset contains features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Euclidean distance between two data points in their computations, this poses to be a problem. If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, 5kg and 5000gms *for illustration*. *The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes. To suppress this effect, we need to bring all features to the same level of magnitudes. This can be achieved by scaling.* 

An illustration : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/scaling.png)

*In a nutshell, scaling helps our optimization algorithm converge faster on our data*. **In the figure we can see the skeweness of the data distribution decreases a lot after scaling, as a result of which gradient descent(optimization algorithm) converges faster**.

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/scalingII.png)

For scaling we will import the **scikit-learn** Python3 machine learning library where we use **MinMaxScaler to scale all the price values beteen 0 and 1, that is the feature range we provided in the code**.

MinMaxScaler in action : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/scalingIII.png)

## (VII) Configuring The Dataset For Deep Learning : 

Since we had planned to use an LSTM model for time-series prediction, the conversion of dataset's shape from 1-D to 3-D tensor became mandatory. For this we grouped the values from the past **time_step** days into one and stacked such units one behind each other.

*Tensor shapes illustration* : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/tensors_shapes.png)

## (VIII) Long Short Term Memory - LSTM : 

Humans don’t start their thinking from scratch every second. As we read this paragraph, we understand each word based on our understanding of previous words. We don’t throw everything away and start thinking from scratch again. Our thoughts have persistence. Traditional neural networks can’t do this, and it seems like a major shortcoming.

**Recurrent Neural Networks(RNNs)** address this issue. They are networks with loops in them, allowing information to persist. 

*A Recurrent Neural Network* :

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/RNN-unrolled.png)

**Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies**.

*A LSTM chain* : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/LSTM3-chain.png)

The entire process of the working behind a RNN is beautifully illustrated at : **https://colah.github.io/posts/2015-08-Understanding-LSTMs/**

Using **Keras API of Tensorflow** a model was prepared having layers of LSTM cells stacked onto each other followed by a general **Artificial Neural Network(ANN)**. 

* **ReLU activation function was used in all the layers with dropout ranging from 0.2-0.4**.
* **Adam Optimizer and MSE(Mean Squared Error) loss function was used.**

Some of our architectures designed using Tensorflow-keras.
*Tesla's architecture* : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/Tesla%20architecture.png)

*AMD's architecture* : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/AMD%20architecture.png)

Once the model was trained we analyzed the loss and predicted prices from the data of the test set(or validation set where test set was not provided).

*Exceprt from AMD's notebook. This was our loss in prediction. As we can clearly see, it's an exponential decrease*.

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/loss%20analysis.png)

# (IX) Results :

**AMD's result** : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/AMD_result.png)

Full view : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/amd_full_view.png)

**Tesla's result** : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/Tesla_result.png)

Full View : 

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/tesla_full_view.png)

# (X) Conclusion : 

The prices were predicted to a considerable accuracy and all this was integrated on a website the team designed using **Django**. 

**Link** : 

Thank you for reading this far !!

![](https://github.com/CodingWitcher/Stock_Investment/blob/master/pics_for_readme/thank-you-1606941_960_720.jpg)
