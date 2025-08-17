# System Requirements:

* Jupyter Notebook([installed](https://jupyter.org/install) or set up)
* Python IDE( ([VS Code ](https://code.visualstudio.com/download)or [PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows#)).
* Python Libraries installed (numpy , pandas, matplotlib , sci-kit learn , seaborn , joblib)

## TASK 1 : MATLAB ML Onramp Course

Gain hands-on experience with Machine Learning fundamentals using MATLAB. This task is designed to introduce you to practical ML workflows through interactive and guided lessons on the MATLAB Machine Learning Onramp Course .

### Your Task:

Enroll in and complete the [MATLAB Machine Learning Onramp](https://matlabacademy.mathworks.com/details/machine-learning-onramp/machinelearning) course.

### Expected Outcomes:

Understand the end-to-end flow of a machine learning project.
Gain exposure to supervised learning techniques in a new tool.
Learn data handling, model training, validation, and performance assessments.
Be able to compare MATLAB’s workflow with Python-based tools like scikit-learn

### Precautions:

Ensure stable internet as your progress might be lost if page resets .

## TASK 2 : Kaggle Crafter — Build & Publish Your Own Dataset

Learn the essentials of data curation, documentation, and publishing by creating and sharing your own dataset on Kaggle. This task will help you understand what makes a dataset usable, discoverable, and valuable to the data science community.
### Your Task:

* Create a dataset of your choice - it can be real or synthetic or fake , but should be cleanly organized.
* Upload the dataset to Kaggle with proper metadata and formatting.
* Your dataset should meet the following usability criteria (total score ≥ 8.5):

### Usability Factors:

#### 1. Completeness

Add a subtitle, tags, description, and a cover image

#### 2. Credibility

* Mention source/provenance (or clarify it’s synthetic
* Add a public notebook demonstrating use (if applicable)
* State the update frequency clearly

#### 3. Compatibility

* Choose a proper license (like CC0 or CC BY)
* Ensure it’s in a compatible format ( .csv , .xlsx , .json , etc.)
* Write clear descriptions for the file and columns

### Expected Outcomes:

* Understand what makes a dataset useful, trustworthy, and user-friendly
* Practice data storytelling through documentation and presentation
* Build visibility on Kaggle with a well-structured contribution
* Gain confidence in preparing and sharing data professionally

### Precautions:

* Avoid sensitive, personal, or plagiarized data
* Keep your dataset clean and minimal

Learn how to create Fake Datasets in Python:
* https://medium.com/@sangitapokhrel911/generating-fake-data-in-python-with-faker-and-drawdata-synthetic-data-generation-dc77d73e8521
* https://youtu.be/xSPJUMpbycA?feature=shared
* https://youtu.be/VJAEMZt_Uh0?feature=shared

Learn how to upload Dataset to Kaggle : 
<br>
[Article](https://medium.com/@sohanaiyappa/how-to-upload-a-dataset-to-kaggle-and-get-a-high-usability-score-a-beginners-guide-b57356a3339d)

## TASK 3 : Data Detox - Data Cleaning using Pandas

Learn how to preprocess and clean raw, messy datasets using Pandas for better machine learning outcomes.
### Your Task:

1. Load the dataset and explore the types of issues present.
2. Handle missing values by either dropping or imputing them.
3. Fix inconsistencies in text or categorical columns (e.g., case mismatches, typos).
4. Format column correctly (e.g., dates as datetime , numbers as int / float ).
5. Remove duplicate rows, if any.
6. Save the cleaned dataset as a new CSV.
### Expected Outcomes:

* Understand real-world data cleaning challenges.
* Gain hands-on experience with Pandas data cleaning methods.
* Learn to prepare data for analysis or modeling.

### Precautions:

* Always inspect the data before applying changes.
* Don’t blindly drop nulls-understand their significance.
* Keep a backup of raw data before applying transformations.

[ Download Dataset](https://drive.google.com/file/d/1GYQK3_S_NWlO0sUIViCPTrVGvdTXvGrC/view?usp=drive_link)

#### Learn Pandas:

* https://youtu.be/mkYBJwX_dMs?feature=shared
* [Official Documentation](https://pandas.pydata.org/docs/)
* [Notebook](https://colab.research.google.com/drive/1h4azbBQEoGC1wyDIXLn3yKCLe2sVssaq?usp=sharing)

### Learn to clean dataset using Pandas:

* https://youtu.be/bDhvCp3_lYw?feature=shared
* https://www.freecodecamp.org/news/data-cleaning-and-preprocessing-with-pandasbdvhj/
* https://www.w3schools.com/python/pandas/pandas_cleaning.asp

## TASK 4 : Anomaly Detection

G-Flix Inc. suspects a breach , but not from the outside. Your job as a **Data Forensics Officer** is to detect unusual patterns in user activity logs using anomaly detection techniques. The twist? You don’t know what the anomaly looks like , it’s hidden in plain sight.

### Your Task:

* Load and explore the provided dataset.
* Identify normal behavior trends using visualizations.
* Apply at least two anomaly detection techniques:
    * Statistical (e.g., Z-score, IQR)
    * Unsupervised ML (e.g., Isolation Forest, DBSCAN)
* Compare flagged anomalies from each method.
* Prepare a final report with your top 5 suspects and evidence.

### Expected Outcomes:

* Understand real-world applications of anomaly detection.
* Gain hands-on experience with unsupervised ML methods.
* Learn how to differentiate between outliers and genuine anomalies.
* Build effective visualizations for behavior profiling.
* Develop investigative storytelling and reporting skills.

### Precautions:

* Scale your data before applying distance-based algorithms.
* Don’t assume every outlier is an anomaly : context is key.
* Use multiple features to justify suspicious behavior.
* Validate anomalies through both visual and algorithmic evidence.

 [Download Dataset](https://drive.google.com/file/d/12B1wpYOWGccgSC-1_BXA5j376xkLO9tr/view?usp=drive_link)
 
#### Understand the concept and implementation:

* https://www.datacamp.com/tutorial/introduction-to-anomaly-detection
* https://youtu.be/UYR5NH_D9g0?feature=shared

#### Learn to implement Different Anomaly Detection Algorithms :

* https://www.geeksforgeeks.org/comparing-anomaly-detection-algorithms-for-outlier-detection-on-toy-datasets-in-scikit-learn/?ref=rp
* https://youtu.be/kN--TRv1UDY?feature=sharedhttps://youtu.be/RDZUdRSDOok?feature=shared

## TASK 5 : Logistic Regression from Scratch

Understand binary classification through hands-on experience by building a logistic regression model from scratch and comparing it with a standard library implementation. The chosen use-case: predicting heart disease.

### Your Task:

* Implement Logistic Regression from Scratch
* Implement Logistic Regression Using scikit-learn
* Compare Models
* Use metrics: accuracy, precision, recall, F1-score
* Discuss:
	* Performance differences
	* Training time
	* Implementation difficulty and interpretability

### Expected Outcomes :

* Master the inner mechanics of logistic regression
* Practice matrix operations and gradient descent
* Learn how scikit-learn abstracts complexity
* Build confidence in choosing the right level of abstraction for ML tasks

## Precautions

* Watch out for issues like vanishing gradients and poor convergence.
* Ensure your dataset is normalized.

[Download Dataset](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data)

Understand Logistic Regression:
* https://youtu.be/yIYKR4sgzI8?feature=shared
* https://www.datacamp.com/tutorial/understanding-logistic-regression-python

Implement Logistic Regression From Scratch:
* https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2/
* https://www.datacamp.com/tutorial/understanding-logistic-regression-python
* https://youtu.be/JDU3AzH3WKg?feature=shared

## TASK 6 : Battle-Test Your Model — Support Vector Machines

Understand and implement Support Vector Machines (SVM) using `scikit-learn` , then stress-test your model by injecting noise into the data to observe how its performance deteriorates. Use Red Wine quality Dataset.

### Your Task:
* Implement SVM Using scikit-learn
* Noise Robustness Experiment:
	* Gradually add Gaussian/random noise to the dataset:
	* Begin with small noise levels (e.g., ±1%)
	* Increase progressively (e.g., ±5%, ±10%, etc.)
* At each level:
	* Retrain the model
	* Evaluate and log the metrics
	* Identify the breakdown point - the level of noise where the model starts to fail.

### Visualize the Results
* Create a line plot of performance metrics vs. noise level
* Highlight the threshold where model performance drops sharply

 [Download Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

### Expected Outcomes:
* Be able to apply SVMs to real-world datasets
* Understand how robust your model is to data corruption
* Gain insight into hyperparameter tuning, model evaluation, and noise handling
### Precautions:
* Check for and handle missing or duplicate data.
* Add noise only to features, not labels.
* Gradually increase noise in small steps (e.g., std dev: 0.01 → 0.5).
* Avoid complex kernels with small datasets.

### Understand SVMs:
* https://youtu.be/8A7L0GsBiLQ?feature=shared
* https://www.geeksforgeeks.org/support-vector-machine-algorithm/
### Implement SVMs:
* https://youtu.be/T9UcK-TxQGw?feature=shared
* https://www.freecodecamp.org/news/svm-machine-learning-tutorial-what-is-the-support-vector-machine-algorithm-explained-with-code-examples/
* https://www.geeksforgeeks.org/support-vector-machine-algorithm/(https://www.geeksforgeeks.org/support-vector-machine-algorithm/)

## TASK 7 : Fairness Meets Functionality

Use the [Utrecht Fairness Recruitment Dataset](https://www.kaggle.com/datasets/ictinstitute/utrecht-fairness-recruitment-dataset) from Kaggle, which contains anonymized recruitment data including age, gender, education, experience, and whether a candidate was hired.

Investigate potential biases in the model by analyzing its predictions across demographic groups such as gender and age. Use fairness metrics like demographic parity and equal opportunity to measure disparities. Discuss any unfair discrimination found and explore possible reasons behind it.
### Your Task:
* Build a Decision Tree from Scratch using ID3 Algorithm
* Evaluate Model Performance
* Use performance metrics:
	* Accuracy
	* Precision
	* Recall
	* F1-score
* Analyze feature importance:
	* Which features were most influential in the tree?
	* Do these make intuitive or ethical sense?
* Conduct a Fairness Analysis
	* Slice the data by demographic groups:
	* Gender (Male/Female/Other)
	* Age brackets (e.g., &lt;25, 25–35, 35+)
### Discuss:
* Demographic Parity: Are hiring decisions independent of gender or age?
* Equal Opportunity: Are qualified candidates from all groups equally likely to be hired?
#### Expected Outcomes:
* Understand decision tree fundamentals and implementation
* Learn how model decisions are formed
* Gain awareness of bias detection techniques in machine learning
* Explore responsible AI practices and ethical model deployment
#### Precautions:
* Limit tree depth and set minimum samples per split.
* Handle missing demographic info appropriately.
* Clearly define demographic groups (e.g., gender, age brackets).
* Watch for proxy variables causing indirect bias.
### Understanding Decision Trees

* https://youtu.be/_L39rN6gz7Y?feature=shared
### Understand ID3 :

* https://youtu.be/YtebGVx-Fxw?feature=shared
* https://youtu.be/CWzpomtLqqs?feature=shared

### Implement ID3 :

- https://towardsdatascience.com/id3-decision-tree-classifier-from-scratch-in-python-b38ef145fd90
- https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f
- https://python-course.eu/machine-learning/decision-trees-in-python.php

## TASK 8 : KNN with Ablation Study

In this task, you'll build a K-Nearest Neighbors (KNN) classifier using the Breast Cancer Wisconsin dataset. The goal is to not only train and evaluate the classifier, but also to conduct a feature ablation study to determine which features are most important for accurate classification. By removing one feature at a time and observing the effect on model performance, you'll identify which features significantly contribute to the model’s prediction.

### Your Task:

*  Preprocess Data:
	* Drop id , `encode diagnosis` (M=1,$ $,B=0)$ , and normalize features.
* Train KNN Model:
	* Use `KNeighborsClassifier` (e.g., $k=5)$ , train-test split, and evaluate.
* Feature Ablation:
	* Remove one feature at a time, retrain, record metrics (accuracy, precision, recall, F1score).
* Analyze Impact:
	* Identify features whose removal drops performance the most.

### Expected Outcomes

* Understand how KNN works and why feature scaling is essential.
* Gain hands-on experience in evaluating model performance.
* Learn how removing features affects model behavior and accuracy.
* Discover which features are most informative in medical datasets.
* Appreciate the value of feature selection and ablation studies in ML pipelines.

### Precautions

* Normalize data before KNN.
* Use consistent k-value for fair comparison.
* Compare all 4 metrics, not just accuracy.
* Ensure only one feature is removed at a time.

#### Understand KNN :

* https://youtu.be/HVXime0nQeI?feature=shared
* https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn

#### Implementation:

* https://medium.com/@amirm.lavasani/classic-machine-learning-in-python-k-nearest-neighbors-knn-a06fbfaaf80a
* https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
* https://youtu.be/rTEtEy5o3X0?feature=shared

## TASK 9 : Evaluation Metrics – Pick the Best Performer!

You will receive 5 pretrained ML models saved as .pkl files. Your goal is to evaluate and compare them using a test dataset and identify the best-performing model.

### Your Task:

* Load the test dataset using pandas .
* Load each pickle file using joblib .
* Use the model to predict on the test set.
* Evaluate:
	* Classification: accuracy, precision, recall, F1-score.
	* Regression: MSE, RMSE, R².
* Compare all scores and conclude which model performs best, with reasons

### Expected Outcomes:

* Learn how to load and evaluate saved models.
* Understand different evaluation metrics and their significance.
* Develop critical analysis by comparing models based on real performance.
* Gain experience in handling multiple model types efficiently.
* Learn to make informed decisions on model selection.

### Precautions:

* Ensure consistent preprocessing (scaling, encoding) between training and testing.
* Check if model type matches the dataset (classification vs regression).
* Handle exceptions if a model fails to load or predict.
* Verify test data shape matches model input requirements.

 [Download Pickle Files](https://drive.google.com/drive/folders/1JohwXYQ3DMUaxgbZPUiVZKMGugQdRqwD) <br>
 [Install Joblib](https://joblib.readthedocs.io/en/latest/installing.html)

Learn how to use Joblib for machine learning:

* https://www.geeksforgeeks.org/save-and-load-machine-learning-models-in-python-with-scikit-learn/
* https://youtu.be/lK0aVny0Rsw?feature=shared
* https://youtu.be/Dm4up8_zJdo?feature=shared

### Evaluation Metrics:

* https://youtu.be/Kdsp6soqA7o?feature=shared
* https://youtu.be/4jRBRDbJemM?feature=shared
* https://youtu.be/LbX4X71-TFI?feature=shared
* https://www.geeksforgeeks.org/metrics-for-machine-learning-model/
* https://medium.com/image-processing-with-python/theoretical-basis-of-ml-model-evaluation-metrics-summary-3cae19129679

