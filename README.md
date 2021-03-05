# supervised_ML_comp118a
Final Project for COGS 118A: An Empirical Comparison of Supervised ML Algorithms across various binary classification problems

## Datasets

All taken from Kaggle

1. Income dataset
	* https://www.kaggle.com/mastmustu/income?select=train.csv
	* 14 features
	* predictor: income_>50k
2. Phishing website detector
	* https://www.kaggle.com/eswarchandt/phishing-website-detector?select=phishing.csv
	* 31 features
	* predictor: 1/-1 phishing website or not
3. Airline passenger satisfaction
	* https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction?select=test.csv
	* 24 features
	* predictor: neutral or dissatisfied / satisfied
4. Surgical Complications dataset
	* https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification
	* 24 features
	* predictor: complication / no complication

## Models

Models I will use for hyperparameter search and classification

1. Logistic Regression
2. SVM
3. Random Forest
4. Artificial Neural Network

## Performance Metrics

1. Accuracy
2. F1 score
3. AUC 
4. Precision
5. Recall
6. Heatmap plots of hyperparameter search results for Logistic Regression, Random Forest, and ANN (SVM has too many hyperparameter combinations over 4 dimensions to show)

