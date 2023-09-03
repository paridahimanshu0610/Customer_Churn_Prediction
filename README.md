# Telco-customer-churn-prediction
This project tackles a classification machine learning challenge focused on predicting customer churn (those who have left the company within the last month, labeled as 'yes' or 'no').

The dataset utilized in this endeavor has been sourced from [Kaggle's Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn). This dataset encompasses various categories of information, such as:

- Details about customers who departed within the previous month, represented by the "Churn" column.
- The array of services each customer has subscribed to, encompassing phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- Customer account particulars, including their tenure, contract type, payment method, paperless billing, monthly charges, and total charges.
- Demographic data pertaining to customers, encompassing gender, and the presence of partners and dependents.

## Methodology

### Exploratory data analysis
1. Visualized count plots for customer churn (target variable) alongside categorical input features, and created histograms for numerical input features.
2. Generated a count plot showcasing customer churn across categories within the categorical input feature.
3. Illustrated numerical feature distributions with box plots, segregated by customer churn.
4. Examined tenure patterns with box plots within various categorical features.
5. Depicted relationships between all numerical features through scatter plots.

### Data Preprocessing
* Removed duplicate rows, handled null values and performed one-hot encoding to encode categorical features. 

### Data Manipulation
* Split data into training and test set with 20% of data being held out for testing.

### Feature Selection
1. Checking for correlation among features and removing correlated features.
2. Used SelectKBest to select the numerical features.

### Feature Scaling
* Performed feature scaling using MinMaxScaler() 

### Addressing Data imbalance
The F1 score for the test set seemed to improve on using a model that had been fitted on the training set transformed using SMOTENC. So, I have used the transformed training dataset to fit my model. 

### Predictive Modelling 
Applied the below models with hyperparameter tuning using GridSearchCV:
1. Logistic Regression
2. Support Vector Classifier
3. KNN
4. Decision Tree
5. Ensemble Methods-Random Forest, AdaBoost, Gradient Boosting, XGBoost
6. ANN
7. Naive Bayes

### Model Evaluation
Used the below metrics to compare the model's performance on test set:
1. Accuracy
2. F1 Score
3. ROC-AUC

### Model Saving and creating app using Flask
* Saved the model as a .pkl file and created an app using Flask that takes user input and predicts the customer churn. 