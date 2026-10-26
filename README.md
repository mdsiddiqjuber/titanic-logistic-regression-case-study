# Titanic Survival Prediction with Logistic Regression

A reproducible case study for **Fundamentals of Data Science** using the supplied Titanic passenger dataset. The project includes exploratory analysis, data preprocessing, logistic-regression training, model evaluation, coefficient interpretation, an executed Jupyter notebook, and a Streamlit prediction app.

## Project structure

```text
.
├── titanic.csv
├── titanic_logistic_regression.ipynb
├── Titanic_Logistic_Regression_Case_Study.pdf
├── Titanic_Logistic_Regression_Case_Study.docx  # editable report
├── analyze.py
├── app.py
├── create_notebook.py  # notebook build utility
├── create_report.py    # editable report build utility
├── requirements.txt
├── README.md
├── figures/          # labelled EDA and evaluation figures
├── outputs/          # metrics, summary statistics, coefficient table
└── models/           # trained preprocessing + logistic-regression pipeline
```

## Results (fixed train/test split)

- Dataset: 891 rows, 12 columns
- Target: `Survived` (0 = did not survive, 1 = survived)
- Split: 80% train / 20% test, stratified, random state 42
- Test accuracy: 79.9%
- Precision: 78.0%
- Recall: 66.7%
- F1-score: 71.9%
- ROC-AUC: 0.844

These values are specific to this preprocessing pipeline and split. They should not be interpreted as guaranteed performance on new data. The dataset has 177 missing Age values, 687 missing Cabin values, and 2 missing Embarked values. The modeling pipeline imputes Age and other numeric inputs with medians, categorical inputs with their most frequent values, scales numeric features, and one-hot encodes categorical features. Cabin, Name, Ticket, and PassengerId are excluded from the model because Cabin is highly incomplete, Name/Ticket are high-cardinality text identifiers in this baseline, and PassengerId is an identifier rather than a passenger characteristic.

## Run locally

Use Python 3.11 or another Python version compatible with the packages below.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python analyze.py
```

Open `titanic_logistic_regression.ipynb` in Jupyter and run all cells to reproduce the analysis and figures. The notebook is already executed and contains outputs from the supplied dataset.

## Run the Streamlit app

After the dependencies are installed and the model has been created by `python analyze.py`:

```bash
streamlit run app.py
```

The app provides inputs for passenger class, sex, age, number of siblings/spouses, number of parents/children, fare, and embarkation port. It reports a model-estimated probability and a binary class at a 0.50 threshold. This is an educational demonstration, not a historical certainty or causal model.

## Publish to GitHub

1. Create a new, empty repository on GitHub (for example, `titanic-logistic-regression-case-study`).
2. Download/extract the project ZIP if needed, open a terminal in this folder, and run:

   ```bash
   git init
   git add .
   git commit -m "Add Titanic logistic regression case study"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/titanic-logistic-regression-case-study.git
   git push -u origin main
   ```

3. Replace `YOUR-USERNAME` with your GitHub username and submit the repository URL with the PDF report. If your course permits a private repository, check that the faculty can access it; otherwise make it public.

## Academic report fields

The report uses editable placeholders for Roll Number, Section, and Faculty Name because those values were not provided. Fill these in before submission.
