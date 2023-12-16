
                                                    

                                           Project Overview
Understanding the factors influencing an individual's decision to obtain a personal loan is critical for strategic decision-making by banks and lending institutions in the dynamic financial services landscape. This project conducts a thorough examination of a large data set containing a wide range of attributes, such as demographic data, financial metrics, and indicators of financial product usage and digital behavior. The primary goal is to decipher the intricate patterns and correlations that contribute to an individual's likelihood of obtaining a personal loan.

                                                         Data Collection 
Data source
 The dataset ‘Bank Loan’ is taken for this project was obtained from Kaggle, a data science and machine learning platform. Kaggle hosts a diverse collection of datasets contributed by the community, encouraging collaboration and exploration in data analysis.

                                                    Data Exploration
ID:
Type: Nominal/Categorical
Description: Unique identifier assigned to everyone.
Age:
Type: Numerical/Continuous
Description: Age of the individual.
Experience:
Type: Numerical/Continuous
Description: Number of years of professional experience.
Income:
Type: Numerical/Continuous
Description: Annual income of the individual.
ZIP Code:
Type: Nominal/Categorical
Description: ZIP Code of the individual's location.
Family:
Type: Numerical/Discrete
Description: Number of family members.
CCAvg:
Type: Numerical/Continuous
Description: Average spending on credit cards per month.
Education:
Type: Ordinal/Categorical
Description: Level of education (e.g., 1: Undergraduate, 2: Graduate).
Mortgage:
Type: Numerical/Continuous
Description: Amount of mortgage debt.
Personal Loan:
Type: Binary/Categorical
Description: Indicates whether the individual accepted a personal loan (1) or not (0).
Securities Account:
Type: Binary/Categorical
Description: Indicates whether the individual has a securities account (1) or not (0).
CD Account:
Type: Binary/Categorical
Description: Indicates whether the individual has a certificate of deposit account (1) or not (0).
Online:
Type: Binary/Categorical
Description: Indicates whether the individual uses online banking services (1) or not (0).
Credit Card:
Type: Binary/Categorical
Description: Indicates whether the individual has a credit card (1) or not (0).

                                                        
                            Exploratory Data Analysis (EDA)

Data Loading
I loaded the dataset from the given URL using the panda's library. The data is read into a Data Frame called ‘Banlloan’ data using the read_csv function. The first few rows of the dataset are then shown using the head () function. 
 

Data Structure 
The dimensions of the ‘Bank Loan’ dataset can be used to summarize its specific structure:
5000 rows (observations) in total
14 (features) in total
Descriptive Statistics
 


Missing Values and Duplicates 
The is.na (). sum () to find the total number of missing values in each column. But there was no null values and duplicates in my dataset.
 



Data Distributions
       The below graph shows the data distribution of age, income, experience, CCavg . Where we can see that data for age, experience is distributed equally but for CC avg, income data is distributed more left  side.
 


 
                Fig-1 (Data distribution for age, experience, income, ccavg )


From the below graph we can see that when we plotted data distribution for income most of the data lies below 100.








 
                                            Fig-2(Data distribution for income)
Data distribution for education column.
 
                                        Fig-3(Data distribution for education column)

Data distribution for online column, we can see that only more that half people are using online banking .
 
                                          Fig-4(Data distribution for online banking usage)
Below graph shows the number of credit card users. We can see that only half people are using credit cards.
 
                              Fig-5(Data distribution for credit card users)
The data distribution for personal loans indicates a significantly low approval rate. It appears to be an imbalanced dataset, which necessitates the implementation of measures during model building to address this imbalance. Failing to do so may result in biased classifications."
 








"When plotting a box plot between personal loan and income, it becomes apparent that the loan approval rate is higher for individuals with an income greater than 100."
 
                             Cleaning and Processing data 

Cleaning and transforming 
Here I am dropping unnecessary columns ('ID','ZIP.Code') from the data set. 
Below table shows the data after dropping the columns id and zip code.
 
Feature scaling 
Since my data set contains numeric values there is no need for dummy variable creation.
"In the dataset, columns such as 'Personal Loan,' 'Security Account,' 'CD Account,' and 'Online' are in binary format. Therefore, normalization is not required for these columns."
Here we are going to normalize only a few columns like 'Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage','Education' so that it will be easy to compare and easy to process for algorithms.
Below two tables show the data before and after normalization.
 Table Before normalization                                         
 
 Table After normalization 
 

Balancing data 
As we mentioned in the previous slides above the value count for class 1 and class 0 in personal.loan columns is unbalanced .Because the number of observations for class 0 is 4494 where number of observations for class 1 is just 480 .
Here I am using one technique called up sampling method to increase the number of observations for class 1 from 480 to 4494 as shown below.
Value count before balance 
0.0    4494
1.0     480
Value count after balance
0.0    4494
1.0    4494



  

                        



                             Machine Learning Model

It's crucial to assess a machine learning model's performance on a dataset it wasn't exposed to during training, to make sure it generalizes well to fresh data. As a result, dividing the available data into training and testing sets is standard procedure. 
The testing set is used to assess the model's performance, and the training set is used to fit the model to the data. We can ascertain whether the model is overfitting to the training data by assessing the model's performance on data that it hasn't been trained on. Overfitting can result in poor generalization performance on new data because the model learns to fit the noise in the training data rather than the underlying patterns.
Training and Testing Sets:
The dataset was divided into two sets for this project. A saved state of random sampling of the dataset will be set for the Training and Testing Datasets in 40% and 60% with random setting as 42 whiles.
Model building 
Here I am using 3 different algorithms to build a model for my data set.
1.Logistic Regression.
2. Decision Tree.
3.Random Forest.

1.	Logistic Regression Model 
A statistical technique called logistic regression is employed in binary classification, meaning that its goal is to forecast the likelihood that an instance will fall into one of two groups. Logistic regression is not a regression algorithm, despite its name. It is a classification algorithm. When the dependent variable is categorical and has only two possible outcomes, such as True or False or Yes or No, 1 or 0 is frequently used. Here I am using Logistic regression because my target variable or my target is in the form of binary format 0, 1 form.

2.	Decision Tree Model 
One popular and adaptable machine learning model that can be used for both regression and classification applications is the decision tree. The model is non-linear and relies on a set of rules to make decisions. The result is a structure resembling a tree, with each internal node denoting a decision based on a feature, each branch representing the decision's result, and each leaf node representing the final prediction or decision. Here the outcome will be class of any type, so if we use this model for our dataset the outcome will be either class1 or class 0.

3.	Random forest Model 
In machine learning, Random Forest is a popular ensemble learning algorithm for tasks involving both regression and classification. It is a member of the tree-based model family and stands out for its strength in handling complex data relationships, minimizing overfitting, and producing reliable predictions. We can use this algorithm to classify our outcome.



Feature and Target Selection:
Here we are using correlation method to select our feature to the target Peersonal.Loan. The target is Personal.loan because we want to analysis on Loan approval.





 
Since variables like 'Income', 'CCAvg', 'Education', 'CD.Account', 'Mortgage','Experience' are highly correlated to the target variable ‘Personal.Loan’ so I am considering these variables as my features .
Features (Independent Variables):
Income:
Type: Numerical/Continuous
Description: The annual income of the individual. This variable represents the financial earnings of the person.


CCAvg:
Type: Numerical/Continuous
Description: Average spending on credit cards per month. This variable gives an indication of the individual's credit card usage and spending habits.

Education:
Type: Ordinal/Categorical
Description: Represents the level of education of the individual. It could be encoded with numerical values corresponding to different education levels (e.g., 1: Undergraduate, 2: Graduate).
CD.Account:
Type: Binary/Categorical
Description: A binary variable indicating whether the individual has a certificate of deposit account (1) or not (0). This variable reflects the presence or absence of a specific financial product.
Mortgage:
Type: Numerical/Continuous
Description: The amount of mortgage debt held by the individual. This variable represents the level of mortgage obligations.
Experience:
Type: Numerical/Continuous
Description: The number of years of professional experience. This variable reflects the individual's work experience over the years.
Target Variable (Dependent Variable):
Personal.Loan:
Whether loan is approved (1) or not (0).
Since we selected features for our model, I want to show the importance of each feature in the model.
 
Model building:
Logistic regression Model:

The Logistic Regression Model is used in this project, and it was trained using the target variable and the chosen column features from the section above.
                     Model = Logistic regression ().

Model Training: 

The model is trained by applying the features 'Income', 'CCAvg', 'Education', 'CD.Account', 'Mortgage','Experience'to the target ‘Personal.Loan’. 

                                           Model evaluation 
Accuracy:
The ratio of correctly predicted instances to all instances is used to calculate accuracy, which is the model's overall correctness. The model's accuracy in this instance is 0.66, or 0.66. This indicates that 66 % of the model's predictions came true.
Confusion matrix:
The model's predictions are broken down in detail in the confusion matrix. The number of false positives (FP), false negatives (FN), true positives (TP), and true negatives (TN) is displayed. This instance:
True Positives (TP) =1577, cases had accurate positive predictions made.
True Negatives (TN) =807, cases were appropriately classified as negative.
False Positives (FP)= 261, cases were mislabeled as positive in advance. 
False Negatives (FN) = 951, cases were mislabeled as negative in advance.

[[1577   261]
 [ 951   807]]


Decision Tree:
 
The Decision Tree Classifier Model is used in this project, and it was trained using the target variable and the chosen column features from the section above.
                      clf = DecisionTreeClassifier(random_state=42)

Model Training: 
The model is trained by applying the features 'Income', 'CCAvg', 'Education', 'CD.Account', 'Mortgage','Experience'to the target ‘Personal.Loan’
                                               Model evaluation 
Accuracy:
With an overall correctness of 97%, the model correctly predicts 97% of the observations it makes.


Confusion Matrix: 
The model's predictions are broken down in detail in the confusion matrix.
True Positives (TP): 4242 cases had accurate positive predictions made.
True Negatives (TN): A total of 4485 cases were appropriately classified as negative.
False Positives (FP): 252 cases were mislabeled as positive in advance.
False Negatives (FN): 9 cases were mislabeled as negative in advance.
                         
[[4242   252]
 [   9   4485]]




Random Forest Classifier:

 The Random Forest Classifier Model is used in this project, and it was trained using the target variable and the chosen column features from the section above.
random_forest = RandomForestClassifier(random_state=42)



Model Training:
 
The model is trained by applying the features 'Income', 'CCAvg', 'Education', 'CD.Account', 'Mortgage','Experience'to the target ‘Personal.Loan’

                                       

                                        
                                            Model evaluation

Accuracy: 
According to the report, the Random Forest model has an accuracy of 95%, meaning that 95% of its predictions come true. For both positive and negative classes, a thorough breakdown of accurate and inaccurate predictions is given by the confusion matrix. To gain a deeper understanding of the model's performance, additional analysis of precision, recall, and other classification metrics can be performed using this data.
The following is displayed in this confusion matrix:

The model's predictions are broken down in detail in the confusion matrix.
True Positives (TP):1681 cases had accurate positive predictions made.
True Negatives (TN): A total of 1749 cases were appropriately classified as negative.
False Positives (FP): 157 cases were mislabeled as positive in advance.
False Negatives (FN): 9 cases were mislabeled as negative in advance.
[[1681   157]
 [   9   1749]]
Since the 3 models give the same accuracy but there is not much difference in model prediction for classification tree model and random forest model. But there is a slight difference in logistic regression model. Among all three models, the decision tree algorithm model performed well.











Performance Matrix:

Metric	 Logistic Regression model	Classification tree model	Random forest Model
Accuracy	0.66	0.97	0.95
 	 	[[4242   252]
 [   9   4485]]
	[[1681   157]
 [   9   1749]]

	 		
Confusion matrix 	[[1577   261]
 [ 951   807]]
		
  	 		
	 		
Precision (Class 0)	
0.62
	
1.00
	
0.99

Recall (Class 0)	
0.86
	
0.94
	
0.91

F1-Score (Class 0)	
0.72      
	
0.97
	
0.95

Precision (Class 1)	
0.76
	
0.95
	
0.92

Recall (Class 1)	
0.46
	
1.00
	
0.99

F1-Score (Class 1)	
0.57      
	
0.97
	
0.95

			


