🧠 Logistic Regression from Scratch
This project implements a Logistic Regression model for binary classification, built entirely from scratch in Python. The model is evaluated on two datasets of varying complexity:

🧍 Small dataset: Titanic survivor prediction (fewer features and instances)

🌦️ Large dataset: Weather prediction in Australia (many features and instances)

📁 Project Structure
bash
Copy
Edit
.
├── logistic_regression.ipynb          # Core logic and model implementation (from scratch)
├── project_classification_small.ipynb  # Classification on the Titanic dataset
├── project_classification_big.ipynb    # Classification on the Australian weather dataset
├── utils.py                           # Utility functions for preprocessing and plotting
⚙️ Workflow for Each Dataset
For both the small and large datasets, the following steps were performed:

📊 Data Visualization – Explore and understand the data

🧹 Data Preprocessing – Clean and prepare the data for training

🔁 Cross-Validation – Tune hyperparameters

(Note: The test set was mistakenly used during validation — oops! 😅)

🧺 Bagging – Apply bootstrap aggregating techniques to improve model robustness

🧪 Model Evaluation – Compare performance of the base logistic regression model vs. the bagged version

📉 Loss Visualization – Plot training loss using evaluate_algorithm(logistic_regression) from the notebook

📂 Notebooks Overview
logistic_regression.ipynb
Implements logistic regression from scratch

Includes training, prediction, regularization (L1/L2), loss calculation, and evaluation logic

project_classification_small.ipynb
Applies logistic regression to the Titanic dataset

Goal: Predict whether a passenger survived the Titanic disaster

project_classification_big.ipynb
Applies logistic regression to the Australian weather dataset

Goal: Predict whether it will rain tomorrow

🧰 Dependencies
Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn (for dataset splitting and evaluation utilities)

📌 Notes
This project was made for educational purposes to understand and implement logistic regression in detail.

No ML libraries were used for modeling — everything is coded from scratch!

🚀 Future Improvements
Fix cross-validation by properly separating a validation set

Experiment with more advanced ensemble methods (e.g., boosting)

Benchmark against library-based implementations for accuracy and speed
