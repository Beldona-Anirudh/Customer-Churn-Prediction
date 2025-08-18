
# Customer Churn Prediction

Customer churn is one of the biggest challenges faced by businesses today. Retaining customers is often more cost-effective than acquiring new ones, so predicting churn can help companies take proactive measures.

This project uses customer data to predict whether a customer is likely to churn or stay. By analyzing customer behavior, demographics, and service usage patterns, the project aims to provide actionable insights to improve retention strategies.



## What I Did

Explored the dataset – 

    Analyzed customer demographics, contract details, payment methods, and service usage.

Visualized churn patterns – 

    Plotted distributions, correlations, and relationships between customer attributes and churn behavior.

Preprocessed the data – 

    Handled missing values, encoded categorical features, and performed scaling where necessary.

Built machine learning models – 

    Trained classifiers (including Random Forest) to predict churn.

Evaluated model performance – 

    Used accuracy, confusion matrix, precision, recall, and F1-score.

Interpreted results – 

    Identified which customer attributes most strongly influence churn.
## Machine Learning Approach

Data Preprocessing

    Converted categorical features (like gender, contract type, payment method) using encoding.

    Split the dataset into training and testing sets.

    Balanced the dataset if churn/no-churn classes were imbalanced.

Modeling

    Random Forest Classifier as the main model.

    Other classifiers (Logistic Regression, Decision Tree, etc.) can be tested for comparison.

Evaluation Metrics

    Accuracy Score → Overall correctness.

    Confusion Matrix → How well the model distinguishes churners from non-churners.

    Precision, Recall, F1-score → Useful for understanding performance on imbalanced data.

    Feature Importance → To identify key drivers of churn.
## Key Insights

Customers on month-to-month contracts are more likely to churn.

Electronic payment methods are often associated with higher churn rates.

Customers with longer tenure are less likely to leave.

Random Forest provided high accuracy and reliable predictions compared to simpler models.
## Tech Stack

-> Python

-> Pandas, NumPy 

    Data wrangling and analysis

->Matplotlib, Seaborn 

    Trend and correlation visualization

-> Scikit-learn 

    Random Forest Classifier, train/test split, evaluation metrics
## How to Run It

Clone the repo

    git clone https://github.com/Beldona-Anirudh/Stock-Price-Movement-Prediction
    


Install the Streamlit

    pip install Streamlit.

Change the Directory

    Switch the cmd directory to the directory where your file is located.

Run 

    Now run the app file with the extension .py .
## Results & Conclusion

The Random Forest Classifier performed best, achieving strong predictive accuracy.

Churn is heavily influenced by contract type, payment method, and tenure.

This model can help businesses take preventive actions such as offering loyalty programs, discounts, or personalized support to at-risk customers.

Future improvements: Try XGBoost, LightGBM, or Neural Networks to further boost performance.
