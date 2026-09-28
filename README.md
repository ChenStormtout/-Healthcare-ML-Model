# 🩺 Early-Stage Diabetes Risk Prediction & Clinical Screening Optimization

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Oversampling-red?style=for-the-badge)](https://imbalanced-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](#)

An end-to-end applied machine learning workflow designed to detect undiagnosed diabetes risk using routine anthropometric and biometric indicators. Built on **5,288 patient records**, this project prioritizes **domain-specific data hygiene**, **class imbalance mitigation**, and **clinical decision threshold calibration** to minimize life-threatening False Negatives.
📌 Executive Summary & Clinical ContextThe Problem: Diabetes mellitus remains one of the most prevalent chronic conditions globally, frequently developing silently until severe microvascular and cardiovascular complications emerge. Routine clinical screening is often hindered by laboratory backlogs and incomplete patient telemetry.The Goal: Build an interpretable, highly sensitive triage classifier that identifies high-risk candidates based on routine non-invasive biometrics, allowing healthcare providers to prioritize patients for confirmatory laboratory testing.Key Outcome: Achieved an ROC-AUC of ~0.95 and ~85%+ Recall using an optimized Random Forest Classifier paired with SMOTE and a calibrated 0.3 classification threshold (halving potential missed diagnoses compared to default 0.5 decision boundaries).🏗 Pipeline Architecture[ Raw Dirty Patient Records (5,288 rows) ]
                    │
                    ▼
[ Domain-Specific Cleansing & Heuristics ]
  ├── Height scale normalization (missing zeros)
  ├── Weight decimal shift handling
  ├── Mathematical BMI recomputation
  └── Biological glucose bounding (40 - 400 mg/dL)
                    │
                    ▼
[ Feature Engineering & Leakage Isolation ]
  ├── WHO BMI Tiers, Age Brackets, BP Comorbidity
  └── Strict isolation of direct diagnostic proxies
                    │
                    ▼
[ Train / Test Split (Stratified 80:20) ]
                    │
                    ▼
[ Data Preprocessing & Scaling (StandardScaler) ]
  └── Scaler fitted STRICTLY on X_train to prevent leakage
                    │
                    ▼
[ Class Imbalance Mitigation (SMOTE) ]
  └── Applied only to training partition
                    │
                    ▼
[ Multi-Model Benchmarking & Threshold Tuning ]
  ├── Logistic Regression | Decision Tree | Random Forest | Gradient Boosting
  └── Threshold shift: 0.5 ──► 0.3 (Prioritizing Sensitivity / Recall)
                    │
                    ▼
[ Clinical Risk Stratification & Triage Protocols ]
🧹 Domain-Specific Data CleansingRaw medical telemetry is notoriously dirty due to manual entry errors, equipment calibration offsets, and delimiter issues. Rather than dropping flawed rows blindly, domain rules were codified:Clinical FeatureIdentified Data AnomalyApplied Domain CorrectionHeight (height)Truncated digits (e.g., 15 or 16 instead of 150–160 cm)Scaled by $10\times$ or $100\times$ based on magnitude ranges ($10 \le x < 100 \implies x \times 10$).Weight (weight)Displaced decimals (e.g., 425 instead of 42.5 kg)Shifted decimal boundaries based on clinical feasibility ($x > 200 \implies x / 10$; $x < 30 \implies x \times 10$).BMI (bmi)Pre-calculated values corrupted by input errorRecomputed mathematically: $\text{BMI} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$ without rounding drift.Glucose (glucose)Extreme noise, unit mismatches, and scale errorsBounded strictly to physiological human thresholds ($40 - 400\text{ mg/dL}$). Values outside were imputed using median.⚙️ Feature Engineering & Leakage Prevention1. Medical Risk StratificationWHO BMI Categories: Grouped continuous BMI into Underweight, Normal, Overweight, and Obese.Age Tiers: Segmented into Young (0–30), Middle (31–45), Senior (46–60), and Elderly (61+) to model non-linear age-related insulin resistance.Comorbid Hypertension: Binary flag (Normal vs. High) tagging patients with systolic $\ge 140\text{ mmHg}$ or diastolic $\ge 90\text{ mmHg}$.2. Preventing Data LeakageDiagnostic Feature Exclusion: glucose_level categories were engineered strictly for Exploratory Data Analysis (EDA) and excluded from the final feature matrix $X$ to avoid circular logic / target leakage.Scaling Isolation: StandardScaler was fitted strictly on the X_train partition and subsequently transformed across X_test to prevent test-set distribution statistics from bleeding into the training process.📊 Exploratory Data Analysis: Key InsightsPrimary Predictor (Glucose): Fasting/random glucose exhibits the highest positive correlation with diabetic status ($r > 0.45$). Patients presenting glucose levels above $200\text{ mg/dL}$ show near-deterministic association with diabetes.Secondary Drivers: BMI and age present strong positive correlations, confirming obesity and metabolic aging as compounding co-factors.Target Imbalance: Non-diabetic patients significantly outnumber diabetic cases, mandating synthetic oversampling via SMOTE on the training set to prevent model apathy toward minority positive cases.🧪 Model Benchmarking & PerformanceFour candidate algorithms were trained on the balanced dataset and evaluated on the untouched, stratified test split ($20\%$ hold-out, $n = 1,058$):Model CandidateAccuracyPrecisionRecall (Sensitivity)F1-ScoreROC-AUCLogistic Regression~0.7820~0.6140~0.7920~0.6910~0.8650Decision Tree~0.8410~0.7100~0.7450~0.7270~0.8320Random Forest (Optimal)~0.8850~0.7420~0.8560~0.7950~0.9520Gradient Boosting~0.8710~0.7280~0.8210~0.7710~0.9380Selected Production Model: Random Forest Classifier (n_estimators=200, max_depth=10, min_samples_split=5, class_weight='balanced').🎯 Clinical Decision Optimization: Why Threshold = 0.3?In standard machine learning classification, the decision threshold defaults to $P(y=1) \ge 0.5$. However, in clinical screening:$$\text{Cost of False Negative (Missed Diabetes)} \gg \text{Cost of False Positive (Unnecessary Lab Test)}$$A False Positive merely results in an inexpensive confirmatory venous blood draw (HbA1c or OGTT).A False Negative leaves a diabetic patient undiagnosed, leading to irreversible kidney damage, neuropathy, or cardiovascular events.Decision Calibration:By systematically adjusting the classification probability threshold from 0.5 down to 0.3:Sensitivity / Recall surged to $>85\%$, capturing borderline patients with moderate glucose elevations coupled with compounding BMI/age risks.The model serves effectively as a high-sensitivity first-line digital triage system.🏥 Actionable Clinical RecommendationsTier-1 Priority Triage: Immediate laboratory referral for individuals with $\text{Glucose} > 200\text{ mg/dL}$, $\text{BMI} \ge 30$, and age $> 45$.Intermediate Borderline Surveillance: Patients scoring in the $0.30 - 0.49$ model probability band (often displaying glucose between $120 - 180\text{ mg/dL}$ with normal weight) should undergo mandatory secondary lifestyle evaluation rather than being dismissed.Primary Healthcare Integration: Deployable as an offline screening assistant in community health centers (Puskesmas) lacking immediate on-site biochemistry laboratories.⚠️ Limitations & GovernanceClinical Limitation: The dataset lacks HbA1c (glycated hemoglobin) measurements, the gold standard for clinical diabetes diagnosis, as well as family hereditary history and longitudinal physical activity data.Synthetic Boundaries: Dataset origin exhibits synthetic regularities; external validation across real-world cohort registries is required prior to bedside clinical deployment.Advisory Disclaimer: This machine learning pipeline is designed exclusively as an auxiliary risk-stratification tool and does not constitute medical diagnosis.📁 Repository Structure├── data/
│   ├── diabetes_dataset_dirty.csv      # Raw input data with formatting errors
│   └── diabetes_cleaned.csv            # Cleaned, validated, and normalized dataset
│
├── notebooks/
│   └── diabetes_prediction_eda_ml.ipynb # Clean, executable Jupyter Notebook
│
├── src/
│   ├── clean_data.py                   # Data cleaning & domain heuristic scripts
│   ├── feature_engineering.py          # Feature transformation functions
│   └── train_evaluate.py               # Model training & threshold evaluation pipeline
│
├── requirements.txt                    # Project dependencies
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
🚀 Quickstart & Reproduction1. Clone & Set Up EnvironmentBashgit clone [https://github.com/ChenStormtout/diabetes-risk-prediction.git](https://github.com/ChenStormtout/diabetes-risk-prediction.git)
cd diabetes-risk-prediction

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Dependencies (requirements.txt)Plaintextpandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
3. Run the PipelineOpen the notebook interactively:Bashjupyter notebook notebooks/diabetes_prediction_eda_ml.ipynb
Or execute the automated training script:Bashpython src/train_evaluate.py
👤 Author & ContactDewa SetyaData Engineer & Analytics SpecialistGitHub: @ChenStormtoutLinkedIn: Dewa SetyaEmail: dewasetya6@gmail.com
