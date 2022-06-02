# aryaAi_assignment
- Data
The assignment was to build a classification model to classify a binary labelled data.
The dataset consist of 3910 rows and 59 columns. The data consist of 57 features(all numeric and anonymous) along with the corresponding class label.

- Exploratory Data Analysis- 
The features were high skewed(right skewed to be specific). 
Tried data transformation of features like log transform/box cox transform etc. to get a better image of the data distribution. Log tranform worked fairly well.
Plotted features against the class labels using boxplots to visualize variation of class labels with the feature values.
Determined spearman correlation coeffecient to identify relation of features with the target variable. 
Plotted heat maps to detrmine correlation between the features. Did not find significant correlation among features.

- Handling Class Imbalance
Checked for class imbalanced. Class-0 was nearly double of class-1. Handeled it using sklearn class_weight parameters.

- Feature Selection
Tried two feature selection  techniques.-1) KbestFeatures using Chi2
                                         2) forward feature selection technique

- Modelling 
- tried several models like Logistic Regression, SVM, Kernel SVM, Xbgoost
- Kernal SVM worked well comapratively and was easier and faster to train.
- Achieved an accuracy of 94.75% f1_score of 92.66% and log_loss of 0.1569 on validation set.

How to run the file

Requirements
list  of dependencies-
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, f1_score
from sklearn.calibration  import CalibratedClassifierCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import chi2, SelectKBest

load the data set on which you want to test the results as a dataframe
Download the pretrained model named finalized_model.sav to print output on test data.
Pass the data to the function final_script in the final_script.ipynb file to generate output.

Example
#install dependencies

- df = pd.read_csv('') 
- svm = pickle.load('') 
- final_script(df, svm)
        
