# Understanding Dataset Structure Through Sample Relationships

## Project Description

This project investigates the internal structure of datasets by analyzing relationships between samples. We explore how sample similarity affects model training and generalization, and investigate whether similarity-aware dataset splitting can improve model performance. This data-centric approach to machine learning reveals hidden patterns in datasets that can inform better model development.

**Why This Project Matters**:
- **Data-Centric ML**: Shifts focus from models to data, revealing how dataset structure affects learning
- **Generalization Insights**: Understanding sample relationships helps predict generalization challenges
- **Splitting Strategies**: Explores whether strategic data splitting can improve model evaluation
- **Hidden Patterns**: Reveals clusters, outliers, and relationships not visible in standard analysis
- **Practical Impact**: Better understanding of data structure leads to better model development

**Key Research Questions**:
- How similar are samples to each other across the dataset?
- How many samples are closely related to a given sample?
- Can dataset splitting be improved using similarity-aware strategies?
- How does sample similarity affect model generalization?
- What patterns emerge in sample relationships?

## Dataset Description

**Dataset Name**: UCI Adult Income Dataset (also known as Census Income Dataset)

**Source**: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/adult)

**Dataset Details**:
- **Number of samples**: ~32,561 (after removing missing values)
- **Number of features**: 14 features
  - **Numerical features (6)**: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - **Categorical features (8)**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Target variable**: income (binary classification)
  - <=50K: Low income
  - >50K: High income
- **Task**: Binary classification (predicting income level)
- **Missing values**: Some features contain '?' which are treated as missing and removed during preprocessing

**Why This Dataset**:
- **Mixed data types**: Numerical and categorical features allow for comprehensive similarity analysis
- **Real-world structure**: Natural clusters and relationships between samples reflect real-world data patterns
- **Classification task**: Binary classification allows clear evaluation of splitting strategies
- **Well-known benchmark**: Standard dataset for ML research ensures reproducibility and comparability
- **Sample diversity**: Diverse feature combinations create interesting similarity patterns to analyze
- **Size**: Large enough (~32K samples) to reveal meaningful patterns in sample relationships

**Data Loading**:
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df = df.dropna()
```

**IMPORTANT**: No synthetic or hard-coded data is used in this project. All experiments use the real UCI Adult Income dataset loaded directly from the UCI repository or OpenML as a fallback.

## Research Questions

1. How similar are samples to each other across the dataset?
2. How many samples are closely related to a given sample?
3. Can dataset splitting be improved using similarity-aware strategies?

## Project Structure

```
project7_dataset_structure_analysis/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_sample_similarity_analysis.ipynb
    ├── 02_strategic_dataset_splitting.ipynb
    └── 03_generalization_comparison.ipynb
```

## Key Experiments

### Experiment A: Sample Relationship Analysis
- Define similarity using Euclidean distance and cosine similarity
- For each sample, count related samples within threshold
- Analyze sample density, clusters, and outliers
- Visualize with heatmaps and distance distributions

### Experiment B: Strategic Dataset Splitting
Compare splitting strategies:
1. Random split (baseline)
2. Stratified split (by target)
3. Similarity-aware split (group similar samples together)

Evaluate:
- Distribution shift between train/test
- Validation accuracy
- Generalization gap

## Learning Objectives

1. Understand dataset structure through similarity analysis
2. Explore the impact of data splitting on generalization
3. Analyze sample relationships and their implications
4. Compare different splitting strategies
5. Develop data-centric ML insights

## Key Insights

- Sample similarity reveals hidden dataset structure
- Similarity-aware splitting can reveal generalization challenges
- Understanding sample relationships informs data collection and preprocessing
- Strategic splitting helps identify potential distribution shifts
