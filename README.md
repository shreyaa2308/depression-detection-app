# Depression Detection Using Social Network Text Analysis
Detecting user-level depression risk using Big Five personality traits, NLP-based speech act classification, and graph-based social influence modeling trained on real Facebook data.

**[Live Demo →](https://depression-detection-application.streamlit.app/)** 

# Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Results](#results)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [References](#references)

# Overview

Depression is one of the most prevalent mental health conditions worldwide. Traditional diagnosis relies heavily on self-reported surveys, which depressed individuals often resist completing. This project proposes a **big data analytics framework** for detecting depression risk at the user level using social network data — without relying on self-reports.

The framework has three core components:

1. **ML Classification** — Seven machine learning models trained on Big Five (OCEAN) personality traits extracted from Facebook status updates to classify users by depression risk
2. **Intention Modeling** — An NLP pipeline using TF-IDF vectorization to classify the speech acts (assertive, directive, expressive) behind user status updates, capturing the *intent* driving social media behavior
3. **Social Influence Modeling** — A graph-based network model using shortest-path algorithms to quantify how depression risk propagates through a user's friendship network

# Dataset

**MyPersonality Dataset** — A real-world Facebook dataset containing:
- Status updates from ~250 users
- Big Five personality trait scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- Depression labels derived from validated psychometric assessments
- Social network connections between users

The MyPersonality dataset is a restricted academic dataset and is not included in this repository. To request access, visit the [MyPersonality Project](http://mypersonality.org/).

An AI-generated speech act dataset is included for the intention modeling component.

# Project Architecture

```
├── 1.Dataset/                        # Dataset files and schema documentation
├── 2.Data_Preprocessing/             # Data cleaning, feature engineering, sentiment scoring
│   └── Output_dataset/               # Cleaned CSV outputs
├── 3.Seven_ML_Model_Implementation/  # 7 classifier notebooks
│   ├── K-Nearest_Neighbors.ipynb
│   ├── Supporting_Vector_Machine.ipynb
│   ├── Neural_Network.ipynb
│   ├── Random_Forest.ipynb
│   ├── CART(decision_tree).ipynb
│   ├── Logistic_Regression.ipynb
│   └── Naive_Bayes.ipynb
├── 4.Intentional_Model/              # TF-IDF + RF/SVM speech act classifier
├── 5.Depression_Social_Metrics/      # Social metric computation notebook
├── 6.Path_Approach/                  # NetworkX graph + shortest-path influence model
└── 7.Tableau_Visualization/          # Tableau workbook + preprocessing notebooks

# Results

# Model Comparison — Depression Risk Classification

All models were trained on Big Five OCEAN trait features with an 80/20 train-test split and evaluated using Precision, Recall, Accuracy, and F1 Score.

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **SVM** ✅ | **74.1%** | **85.1%** | 74.9% | 98.3% |
| Neural Network | 73.5% | 84.6% | 76.3% | 94.9% |
| Random Forest | 70.7% | 82.3% | 75.4% | 90.5% |
| KNN | 69.5% | 81.5% | 74.9% | 89.3% |
| Logistic Regression | 68.8% | 81.0% | 74.6% | 88.6% |
| Naive Bayes | 64.4% | 77.3% | 74.3% | 80.5% |
| CART (Decision Tree) | 59.6% | 71.8% | 75.7% | 68.2% |

**SVM achieved the best overall performance** with 74.1% accuracy and 85.1% F1 score — selected as the final model for deployment.

### Intention Model — Speech Act Classification
- Classified user status updates into speech acts: **assertive, directive, expressive, commissive**
- Used TF-IDF vectorization + Random Forest and SVM classifiers
- 80% training / 20% test split with 10-fold cross-validation on training set

### Social Influence Model — Network Graph
- Built a directed friendship network of 250 users using **NetworkX**
- Applied **shortest-path algorithms** to compute influence propagation scores
- Quantified how a user's depression risk is affected by their connections' mental states
- Visualized influence scores and network topology in Tableau


## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib networkx jupyter
```

### Step 1 — Data Preprocessing
```bash
cd 2.Data_Preprocessing
jupyter notebook Data_preprocessing.ipynb
```

### Step 2 — Run ML Models
```bash
cd 3.Seven_ML_Model_Implementation
jupyter notebook Supporting_Vector_Machine(SVM).ipynb   # Best performing model
```
Each notebook is self-contained. Run any model independently by loading `mypersonality_cleaned.csv` from the Dataset folder.

### Step 3 — Intention Model
```bash
cd 4.Intentional_Model
python intention_modelling.py
```

### Step 4 — Social Influence / Path Approach
```bash
cd 6.Path_Approach
jupyter notebook PATH_PATTERN_CODE.ipynb
```

### Step 5 — Tableau Dashboard
Open `7.Tableau_Visualization/Depression_Network_Analysis.twb` in Tableau Desktop.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML & Modeling | Scikit-Learn (SVM, KNN, Random Forest, Neural Network, CART, Logistic Regression, Naive Bayes) |
| NLP | TF-IDF Vectorization, Speech Act Classification |
| Network Analysis | NetworkX, Shortest-Path Algorithm |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Tableau |
| Environment | Jupyter Notebook, Google Colab |

---

## Key Findings

- **Personality traits alone are meaningful predictors of depression risk** — SVM achieved 74.1% accuracy using only Big Five OCEAN scores as features
- **High recall (98.3% for SVM) is critical** in mental health contexts — it is preferable to flag more users for review than to miss at-risk individuals
- **Social influence propagates depression risk** — users with 2-3 degrees of connection to depressed users showed measurably higher influence scores in the network model
- **Speech act patterns differ** between depressed and non-depressed users — depressed users showed higher rates of expressive and directive speech acts

---

## References

1. Yang, X., McEwen, R., Ong, L. R., & Zihayat, M. (2020). A big data analytics framework for detecting user-level depression from social networks. *International Journal of Information Management*, 54, 102141.
2. Adamopoulos, P., Ghose, A., & Todri, V. (2018). The impact of user personality traits on word of mouth: Text-mining social media platforms. *Information Systems Research*, 29(3), 612–640.

---

## Ethics Note

This project is academic in nature and intended to support mental health research. The dataset used contains anonymized user data collected under informed consent for research purposes. Depression risk scores produced by this model are **not diagnostic** and should not replace professional clinical assessment.
