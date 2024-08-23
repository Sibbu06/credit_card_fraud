Credit Card Fraud Detection Project
Overview
This project aims to build a machine learning model to detect fraudulent credit card transactions. Using a dataset of transactions made by European cardholders in September 2013, the goal is to accurately identify fraudulent transactions while minimizing false positives. The project explores various data preprocessing techniques, feature engineering, and machine learning models to achieve high accuracy and robustness.

Dataset
The dataset consists of 284,807 transactions, of which 492 are classified as fraudulent. The dataset is highly imbalanced, with fraudulent transactions making up only 0.17% of the total data. The features include time of transaction, amount, and 28 anonymized variables (V1, V2, ..., V28) resulting from a PCA transformation.

Project Workflow
1. Data Exploration
Initial exploration of the dataset to understand the distribution of features and the imbalance in the target variable.
Visualization of transaction activity over time, focusing on the times of day when transactions occur most frequently.

2. Feature Engineering
Datetime Features: Extracted hour, minute, and second from the Time feature and created cyclic features (Hour_sin, Hour_cos) to capture the periodic nature of transactions.
Day Phases: Introduced a Day_Phases feature to categorize the day into four phases: Morning, Afternoon, Evening, and Night. This helped identify patterns in transaction activity throughout the day.
Scaling: Applied StandardScaler to numerical features to standardize them, ensuring that all features were on the same scale and aiding model performance.

3. Dimensionality Reduction
Applied Principal Component Analysis (PCA) to reduce the number of features from 31 to 13 components based on Kaiser’s Criterion. This helped to retain the most significant information while reducing the complexity of the model.

4. Model Selection and Training
Random Forest Classifier:
Chose a Random Forest model due to its ensemble nature and robustness in handling imbalanced datasets.
Evaluated the initial model, which produced perfect scores of 1.00 for precision, recall, and F1-score. This raised concerns about potential overfitting or data leakage.

Hyperparameter Tuning:
Tuned the Random Forest model using GridSearchCV to find the optimal hyperparameters, specifically n_estimators, to improve the model's performance.
The tuned model achieved an accuracy of 99.96%, with not much improvement.

5. Deep Learning Model (Comparison)
Implemented a deep learning model using TensorFlow and Keras to compare its performance against the RandomForestClassifier.
The model consisted of multiple Dense layers with ReLU activation functions, Dropout layers for regularization, and a softmax output layer.
Achieved a test accuracy of 98.90% and a test loss of 0.0301, indicating good generalization.

6. Model Evaluation
Used confusion matrix and classification report to evaluate the models on precision, recall, F1-score, and accuracy.
Both the Random Forest and deep learning models showed excellent performance, with the Random Forest slightly outperforming in terms of accuracy.

7. Model Saving
Saved the trained models using the joblib library, ensuring they can be easily loaded for future use in new projects.
Conclusion
The project successfully developed and fine-tuned a machine learning model capable of accurately detecting fraudulent credit card transactions. The use of PCA for dimensionality reduction, combined with advanced ensemble methods and deep learning, provided robust and reliable models. Despite the highly imbalanced dataset, the final models achieved near-perfect accuracy, demonstrating the effectiveness of the approach.

Future Work
Cross-Validation: Implement cross-validation techniques to further ensure the robustness of the model.
Anomaly Detection: Explore unsupervised learning methods for anomaly detection to complement the supervised approaches.
Real-Time Implementation: Consider deploying the model in a real-time environment to detect fraud as transactions occur.
