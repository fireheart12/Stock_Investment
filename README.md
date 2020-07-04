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

