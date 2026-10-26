from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'titanic.csv'
FIG_DIR = BASE_DIR / 'figures'
MODEL_DIR = BASE_DIR / 'models'
OUT_DIR = BASE_DIR / 'outputs'
for folder in (FIG_DIR, MODEL_DIR, OUT_DIR): folder.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TARGET = 'Survived'
NUMERIC_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare']
CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

sns.set_theme(style='whitegrid', context='notebook')

# 1. Load and validate data
df = pd.read_csv(DATA_PATH)
required = set(FEATURES + [TARGET])
missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f'Missing required columns: {sorted(missing_columns)}')

# 2. EDA summary outputs
summary = df.describe(include='all').transpose()
summary.to_csv(OUT_DIR / 'descriptive_statistics.csv')
df.isna().sum().rename('MissingCount').to_frame().assign(
    MissingPercent=lambda x: (x['MissingCount'] / len(df) * 100).round(2)
).to_csv(OUT_DIR / 'missing_values.csv')

# Figure 1: target distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df[TARGET].value_counts().sort_index()
ax.bar(['Did not survive (0)', 'Survived (1)'], [counts.get(0,0), counts.get(1,0)])
ax.set_title('Titanic Passenger Survival: Class Distribution')
ax.set_xlabel('Outcome')
ax.set_ylabel('Number of passengers')
for i, value in enumerate([counts.get(0,0), counts.get(1,0)]): ax.text(i, value + 8, f'{value} ({value/len(df):.1%})', ha='center')
fig.tight_layout(); fig.savefig(FIG_DIR/'01_survival_distribution.png', dpi=180); plt.close(fig)

# Figure 2: Survival percentage by sex
sex_rates = df.groupby('Sex', observed=False)[TARGET].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(sex_rates.index.astype(str), sex_rates.values * 100)
ax.set_title('Observed Survival Rate by Sex')
ax.set_xlabel('Sex'); ax.set_ylabel('Passengers who survived (%)'); ax.set_ylim(0,100)
for i,v in enumerate(sex_rates.values*100): ax.text(i,v+2,f'{v:.1f}%',ha='center')
fig.tight_layout(); fig.savefig(FIG_DIR/'02_survival_by_sex.png', dpi=180); plt.close(fig)

# Figure 3: Survival percentage by class
class_rates = df.groupby('Pclass', observed=False)[TARGET].mean().sort_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(class_rates.index.astype(str), class_rates.values * 100)
ax.set_title('Observed Survival Rate by Passenger Class')
ax.set_xlabel('Passenger class (1 = first, 3 = third)'); ax.set_ylabel('Passengers who survived (%)'); ax.set_ylim(0,100)
for i,v in enumerate(class_rates.values*100): ax.text(i,v+2,f'{v:.1f}%',ha='center')
fig.tight_layout(); fig.savefig(FIG_DIR/'03_survival_by_class.png', dpi=180); plt.close(fig)

# Figure 4: Age distributions by survival label
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.histplot(data=df, x='Age', hue=TARGET, bins=25, stat='density', common_norm=False, element='step', fill=False, ax=ax)
ax.set_title('Passenger Age Distribution by Survival Outcome')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Density')
fig.tight_layout(); fig.savefig(FIG_DIR/'04_age_distribution_by_survival.png', dpi=180); plt.close(fig)

# Figure 5: Missingness
miss_pct = df.isna().mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.8))
plot_miss = miss_pct[miss_pct > 0]
ax.bar(plot_miss.index, plot_miss.values)
ax.set_title('Missing Data by Column')
ax.set_xlabel('Dataset column'); ax.set_ylabel('Missing values (%)')
ax.tick_params(axis='x', rotation=35)
for i,v in enumerate(plot_miss.values): ax.text(i, v+1.2, f'{v:.1f}%',ha='center',fontsize=9)
ax.set_ylim(0, max(plot_miss.values)*1.15)
fig.tight_layout(); fig.savefig(FIG_DIR/'05_missing_values.png', dpi=180); plt.close(fig)

# Figure 6: correlation matrix of numerical data
corr_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
fig, ax = plt.subplots(figsize=(8.2, 6.5))
sns.heatmap(df[corr_cols].corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Correlation Matrix for Selected Numeric Variables')
fig.tight_layout(); fig.savefig(FIG_DIR/'06_correlation_heatmap.png', dpi=180); plt.close(fig)

# 3. Split before fitting preprocessing to avoid leakage
X = df[FEATURES].copy()
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, NUMERIC_FEATURES),
    ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])
model.fit(X_train, y_train)

# 4. Evaluation
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    'dataset_rows': int(len(df)),
    'dataset_columns': int(df.shape[1]),
    'training_rows': int(len(X_train)),
    'testing_rows': int(len(X_test)),
    'survivors': int(y.sum()),
    'non_survivors': int((y == 0).sum()),
    'survival_rate': float(y.mean()),
    'accuracy': float(accuracy_score(y_test, pred)),
    'precision': float(precision_score(y_test, pred, zero_division=0)),
    'recall': float(recall_score(y_test, pred, zero_division=0)),
    'f1_score': float(f1_score(y_test, pred, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, proba)),
    'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
    'classification_report': classification_report(y_test, pred, output_dict=True, zero_division=0),
    'random_state': RANDOM_STATE,
    'test_size': 0.20,
    'features': FEATURES,
}
with open(OUT_DIR/'metrics.json','w') as f: json.dump(metrics, f, indent=2)
pd.DataFrame([{'Metric':k,'Value':metrics[k]} for k in ['accuracy','precision','recall','f1_score','roc_auc']]).to_csv(OUT_DIR/'model_metrics.csv', index=False)

# Figure 7: ROC curve
fpr, tpr, thresholds = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(7, 5.3))
ax.plot(fpr, tpr, label=f'Logistic regression (AUC = {metrics["roc_auc"]:.3f})', linewidth=2)
ax.plot([0,1],[0,1], linestyle='--', label='Random classifier (AUC = 0.500)')
ax.set_title('ROC Curve: Titanic Survival Prediction')
ax.set_xlabel('False Positive Rate (1 - Specificity)'); ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(FIG_DIR/'07_roc_curve.png', dpi=180); plt.close(fig)

# Figure 8: Confusion matrix
cm = confusion_matrix(y_test, pred, labels=[0,1])
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Predicted not survived','Predicted survived'],
            yticklabels=['Actual not survived','Actual survived'])
ax.set_title('Confusion Matrix: Held-Out Test Set')
ax.set_xlabel('Predicted class'); ax.set_ylabel('Actual class')
fig.tight_layout(); fig.savefig(FIG_DIR/'08_confusion_matrix.png', dpi=180); plt.close(fig)

# Figure 9: Coefficients / odds ratios
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefs = model.named_steps['classifier'].coef_[0]
coef_df = pd.DataFrame({'Feature':feature_names, 'Coefficient':coefs})
coef_df['Odds ratio'] = np.exp(coef_df['Coefficient'])
coef_df['Absolute coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Coefficient')
coef_df.to_csv(OUT_DIR/'model_coefficients.csv', index=False)
fig, ax = plt.subplots(figsize=(9,6))
colors = ['#c0504d' if x < 0 else '#4f81bd' for x in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Logistic Regression Coefficients (Model Feature Space)')
ax.set_xlabel('Coefficient (log-odds scale)'); ax.set_ylabel('Encoded feature')
fig.tight_layout(); fig.savefig(FIG_DIR/'09_model_coefficients.png', dpi=180); plt.close(fig)

# Save model and metadata
joblib.dump(model, MODEL_DIR/'titanic_logistic_regression_pipeline.joblib')
metadata = {
    'features': FEATURES, 'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES, 'target': TARGET,
    'random_state': RANDOM_STATE, 'test_size': 0.2,
    'threshold': 0.5, 'sklearn_version': __import__('sklearn').__version__
}
with open(MODEL_DIR/'metadata.json','w') as f: json.dump(metadata, f, indent=2)

# Compact result summary for build/report
group_findings = {
    'survival_rate_by_sex': {str(k): float(v) for k,v in df.groupby('Sex')[TARGET].mean().items()},
    'survival_rate_by_class': {str(k): float(v) for k,v in df.groupby('Pclass')[TARGET].mean().items()},
    'missing_age_count': int(df['Age'].isna().sum()),
    'missing_cabin_count': int(df['Cabin'].isna().sum()),
    'missing_embarked_count': int(df['Embarked'].isna().sum()),
    'median_age': float(df['Age'].median()),
    'median_fare': float(df['Fare'].median()),
    'features_excluded': ['PassengerId', 'Name', 'Ticket', 'Cabin'],
}
with open(OUT_DIR/'statistical_findings.json','w') as f: json.dump(group_findings, f, indent=2)

print('DATASET SHAPE:', df.shape)
print('TARGET COUNTS:', y.value_counts().sort_index().to_dict())
print('SURVIVAL RATE BY SEX:', group_findings['survival_rate_by_sex'])
print('SURVIVAL RATE BY CLASS:', group_findings['survival_rate_by_class'])
print('MISSING VALUES:', df.isna().sum().to_dict())
print('TRAIN / TEST:', len(X_train), '/', len(X_test))
print('METRICS:', {k: round(metrics[k],4) for k in ['accuracy','precision','recall','f1_score','roc_auc']})
print('CONFUSION MATRIX [true rows 0,1; predicted columns 0,1]:', cm.tolist())
print('TOP POSITIVE COEFFICIENTS:', coef_df.nlargest(5,'Coefficient')[['Feature','Coefficient','Odds ratio']].round(3).to_dict('records'))
print('TOP NEGATIVE COEFFICIENTS:', coef_df.nsmallest(5,'Coefficient')[['Feature','Coefficient','Odds ratio']].round(3).to_dict('records'))
print('Model saved to:', MODEL_DIR/'titanic_logistic_regression_pipeline.joblib')
