# Model Card — HR Job-Change Classifiers

## Intended use

This project is an educational comparison of configurable traditional and deep-learning pipelines for tabular binary classification. It may support research demonstrations and supervised analysis of the public HR Analytics dataset.

It must not be used to automatically make hiring, termination, promotion, compensation, or employee-retention decisions. Any real-world use would require a new legal, privacy, fairness, and external-validation review with human oversight.

## Models and decision protocol

- Traditional: Logistic Regression and Random Forest.
- Deep learning: tabular ResNet and TabM.
- Preprocessing and category vocabularies are fit on train only.
- Candidates and thresholds are selected on validation PR-AUC/F1.
- The held-out test set is used for one locked evaluation. Later bootstrap, calibration, and subgroup analyses are descriptive and do not select a new champion.

## Data and limitations

The dataset contains 19,158 self-reported candidate profiles. The positive label records stated job-change interest rather than a realized employment outcome. Missingness, sampling, historical practices, and self-selection may encode collection-process bias. Results from one public dataset should not be assumed to transfer to another country, occupation, employer, or time period.

Gender is incomplete and unevenly represented. The subgroup audit reports error rates for sufficiently large observed groups, including a separate missing group; it does not establish legal or substantive fairness. Feature importance describes predictive association, not causality.

## Evaluation and uncertainty

The project reports Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, and PR-AUC. For the imbalanced target, PR-AUC and F1 are primary. It also reports repeated cross-validation for locked traditional configurations, paired bootstrap intervals, Brier score/ECE, validation ablations, permutation importance, and gender subgroup error rates.

The observed F1 improvements of the deep models over Random Forest have confidence intervals that include zero. ResNet's paired PR-AUC improvement is positive at the reported 95% bootstrap interval, while TabM's is not. These are estimates on the same finite test sample, not proof of general superiority.

## Reproducibility

The repository includes split indices, preprocessing artifacts, model state, experiment configurations, locked predictions, package versions, and SHA-256 checksums. See `reports/audit_provenance.json`. A clean Colab run should reproduce all generated outputs before submission.
