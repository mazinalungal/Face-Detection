# Password Strength Prediction using Machine Learning

## 1. Problem Statement
Password security is a major issue in modern digital systems. Weak passwords are easy to guess and can lead to account breaches, identity theft, data loss, and financial harm. This project builds a machine learning model to classify passwords as weak, medium, or strong based on structural features such as length, uppercase and lowercase counts, digits, symbols, repetition, and entropy. The goal is to identify risky passwords before they are used in real systems.

## 2. Dataset Description
The project uses a synthetic password dataset created from password-strength patterns to simulate real-world conditions. The final dataset contains more than 3,000 rows and several engineered features. The target column is `strength`, which has three classes: weak, medium, and strong. Each row represents one password and the characteristics used to distinguish it.

The main features are:
- length
- uppercase count
- lowercase count
- digits count
- symbols count
- unique_chars
- repeated_chars
- common_word
- has_upper
- has_lower
- has_digit
- has_symbol
- is_common_pattern
- entropy
- pattern_family

## 3. Preprocessing
The dataset was checked for missing values and duplicate entries. Missing values were removed and duplicates were dropped to avoid biased training. Text features such as `pattern_family` were converted into a numeric format using one-hot encoding. For model stability, numerical features were scaled using `StandardScaler` inside a `Pipeline` to avoid data leakage between training and testing data. The dataset was then split into an 80% training set and a 20% testing set with stratification to maintain class balance.

## 4. Algorithm Selection
Three classifiers were compared:
- Logistic Regression: used as a linear baseline; simple, fast, and interpretable.
- Support Vector Machine (SVM): useful for decision boundaries in high-dimensional feature spaces.
- Random Forest: chosen as the main model because it handles nonlinear patterns well and is robust on structured tabular data.

The final model was selected using 5-fold cross-validation and the best model was chosen based on the highest F1 score, because F1 balances precision and recall well for classification tasks.

## 5. Training and Testing
The models were trained on the training set using `.fit()`, and predictions were generated on the test set using `.predict()`. Metrics were then calculated for the final model, including accuracy, precision, recall, F1-score, and ROC-AUC. A confusion matrix was also generated to identify where the model made mistakes.

## 6. Evaluation Metrics
The model evaluation table shows the comparison of the machine learning models. The final selected model performed best under cross-validation. The test-set metrics were:
- Accuracy: high confidence for correct classification over all classes
- Precision: measure of how many predicted strong passwords were actually strong
- Recall: measure of how many actual strong passwords were correctly found
- F1-score: combined measure for precision and recall
- ROC-AUC: effectiveness in distinguishing between classes

## 7. Graphs
The project generated several plots and saved them to the `plots` folder.
- Class distribution chart
- Correlation heatmap
- Confusion matrix
- ROC curve
- Model comparison bar chart

These visualizations help explain the behavior of the model and show whether the classes are balanced or whether specific strength categories are harder to classify.

## 8. Analysis
The Random Forest model won because it handled the non-linear patterns in password features better than the linear baseline and had better overall F1 performance than the other models. Misclassifications usually happened between neighboring classes, such as medium and strong passwords, which reflects how some password patterns are naturally ambiguous.

The most important metric depends on the application. For a real security system, recall is often more important because a missed weak password is riskier than a false alarm. However, precision is also important to avoid mistakenly labeling ordinary passwords as weak when they are not. The final choice balanced both metrics, and the model was selected using F1-score because it provides a strong overall evaluation for this type of classification problem.

## 9. Limitations
This project uses a synthetic password dataset rather than a large real-world password dataset. That means it may not capture the full complexity of real password usage, including attacker patterns, leaked-password distributions, and password reuse patterns. In addition, some generated passwords may look more formulaic than real user-created passwords. Future work could use a larger real dataset and add features such as common dictionary words, keyboard patterns, and breach-password matching.

## 10. Mapping to Course Outcome
This project satisfies CO5: "Solve real life problems using appropriate machine learning models and evaluate the performance measures".

The project follows a complete machine learning workflow:
- Problem identification
- Data preparation and preprocessing
- Model comparison and selection
- Training and testing
- Metric evaluation
- Visualization and interpretation

This demonstrates the use of machine learning to solve a practical security problem with measurable model performance.

## 11. Conclusion
The password strength prediction model successfully classifies passwords into weak, medium, and strong categories using engineered features and machine learning. The Random Forest model was the strongest performer, and the evaluation metrics show that the chosen model is suitable for a real-world prototype. The project also includes a Streamlit app so users can enter a password and receive a prediction instantly.
