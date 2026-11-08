# ML Assignment Completion Checklist

Last audited: 2026-11-05

Use this file to track the remaining work before submission. Checked items were verified in the repository during the audit. Items marked **External confirmation** cannot be verified from repository files.

## 1. Dataset and problem definition

- [x] Use a tabular dataset with a sufficiently large sample size (19,158 rows).
- [x] Include categorical variables.
- [x] Include missing values that require preprocessing.
- [x] Define and document the binary classification target.
- [x] Document the public Kaggle dataset source in `README.md`.
- [x] Add the Kaggle dataset source link and a short dataset description directly to the final notebook.
- [ ] Confirm that no other class group uses the same dataset. **External confirmation**
- [ ] Confirm that the dataset/topic has been approved or discussed with the lecturer. **External confirmation**

## 2. Exploratory data analysis

- [x] Inspect dataset shape, columns, types, and descriptive statistics.
- [x] Analyze missing values.
- [x] Analyze numerical feature distributions.
- [x] Analyze categorical feature distributions.
- [x] Analyze the target distribution and class imbalance.
- [x] Analyze correlations and relationships with the target.
- [x] Save EDA figures under `reports/figures/`.
- [x] Summarize the important EDA findings in the report PDF.

## 3. Data preparation and extracted features

- [x] Separate train, validation, and test data.
- [x] Prevent preprocessing leakage by fitting transformations on training data only.
- [x] Impute missing numerical and categorical values.
- [x] Encode categorical features.
- [x] Implement feature engineering.
- [x] Implement StandardScaler and MinMaxScaler configurations.
- [x] Implement PCA configurations with 90% and 95% explained variance.
- [x] Save train, validation, and test feature arrays as `.npy` files.
- [x] Save target arrays, preprocessing metadata, split indices, and fitted preprocessors.
- [x] Keep extracted features under `features/`.

## 4. Traditional machine-learning pipeline

- [x] Implement Logistic Regression.
- [x] Implement Random Forest.
- [x] Make preprocessing steps, model parameters, and experiment configurations configurable.
- [x] Compare multiple preprocessing and hyperparameter configurations.
- [x] Address class imbalance with class weights.
- [x] Select classification thresholds using validation data only.
- [x] Evaluate accuracy, balanced accuracy, precision, recall, specificity, F1, ROC-AUC, and PR-AUC.
- [x] Generate confusion matrices, ROC curves, and precision-recall curves.
- [x] Save validation and final test results under `reports/`.
- [x] Save trained traditional models under `models/`.
- [x] Explain the parameter choices and configuration comparison in the report PDF.

## 5. Deep-learning extension

- [x] Implement a residual MLP (ResNet) baseline suitable for tabular data.
- [x] Implement TabM as a parameter-efficient internal ensemble using shared weights and rank-one adapters.
- [x] Implement training, validation, early stopping, checkpointing, and threshold selection.
- [x] Tune three candidates per deep-learning algorithm using validation PR-AUC only.
- [x] Compare both deep models against both traditional models on the same test split.
- [x] Improve test F1 over Random Forest: ResNet `0.6438`, TabM `0.6427`, Random Forest `0.6389`.
- [x] Improve test PR-AUC over Random Forest: ResNet `0.5657`, TabM `0.5576`, Random Forest `0.5414`.
- [x] Explain why F1 and PR-AUC are the primary comparison metrics for the imbalanced target.
- [x] Clearly disclose that Random Forest retains the highest ROC-AUC and that the deep models do not improve every metric.
- [x] Rewrite the LaTeX report to replace the previous FT-Transformer experiment with ResNet and TabM results.
- [x] Add repeated CV, validation ablation, paired bootstrap, calibration, permutation importance, subgroup audit, and a model card.

## 6. Notebook and Google Colab

- [x] Provide `notebooks/ML_Tabular_Final.ipynb` as the main frontend.
- [x] Keep the complete EDA, traditional ML, and deep-learning workflows accessible from the notebook.
- [x] Avoid mounting personal Google Drive storage.
- [x] Save an executed notebook with no saved cell errors.
- [x] Add a Google Colab badge to `README.md`.
- [x] Add an automatic dependency-installation step for `requirements.txt` in Colab.
- [x] Make public data acquisition explicit and automatic in the notebook, with SHA-256 validation.
- [ ] Commit and push every file required by the notebook to the branch used by Colab.
- [x] Replace the working `feature/s1-final` Colab link/clone branch with `main`; change to the immutable submission tag at release time.
- [ ] Open a fresh Colab runtime and verify **Runtime -> Run all** without manual intervention.
- [ ] Confirm all expected feature, model, figure, and result files are produced in the fresh runtime.

## 7. Automated verification

- [x] Add automated tests under `tests/`.
- [x] Verify the test suite passes (current audit: 11/11 tests passed, including the notebook contract).
- [ ] Rerun the complete test suite immediately before submission.
- [ ] Record the final test command and result in the report or submission notes.

## 8. Required folder structure and deliverables

- [x] Provide `notebooks/`.
- [x] Provide reusable code under `modules/`.
- [x] Provide experiment outputs under `reports/`.
- [x] Provide extracted arrays under `features/`.
- [x] Provide experiment configurations under `configs/`.
- [x] Provide saved artifacts under `models/`.
- [x] Provide `requirements.txt`.
- [x] Create the Vietnamese project report PDF under `reports/` (`report_vi.pdf`, currently containing clearly marked team-information placeholders).
- [x] Ensure the report contains EDA, pipeline design, configurations, experiments, comparisons, analysis, conclusions, limitations, and future improvements.
- [x] Add the report PDF and LaTeX source links to `README.md`.
- [ ] Create the final submission archive/package if required by the LMS instructions.

## 9. Team contribution requirements

- [x] Add a detailed work-assignment table template to the report.
- [ ] Include each member's full name and student ID in the table.
- [ ] List each member's assigned tasks and deliverables.
- [ ] Include a completion/contribution percentage for each member.
- [ ] Confirm that the percentages follow the grading formula in the assignment specification.
- [ ] Add meeting minutes, screenshots/photos, or other teamwork evidence if claiming the teamwork evidence marks.
- [ ] Organize teamwork evidence, for example under `reports/team_evidence/`.

## 10. README and repository metadata

- [x] State the course name, semester, and academic year.
- [x] Describe the project objective and dataset.
- [x] Provide local installation and execution instructions.
- [x] Document the library requirements.
- [x] Describe the repository structure.
- [x] Summarize the traditional and deep-learning results.
- [ ] Replace the missing course-code placeholder.
- [ ] Replace the missing lecturer-name placeholder.
- [ ] Add the missing student IDs.
- [ ] Add every member's email address.
- [x] Add the report PDF link.
- [ ] Update the Colab badge to the final branch or release tag.
- [ ] Remove all `Chưa được cung cấp`, `Đang cập nhật`, and similar submission placeholders.

## 11. Team and administrative confirmation

- [x] Record the team's confirmation that the lecturer approved the four-person exception; attach the actual written evidence before submission. **External evidence still required**
- [ ] Confirm that the team is registered in the LMS Google Sheet. **External confirmation**
- [ ] Confirm that all required periodic progress reports were submitted. **External confirmation**
- [ ] Check the final deadline and submission instructions in the LMS. **External confirmation**
- [ ] Upload the final package to the lecturer's required Drive/LMS location. **External confirmation**
- [ ] Verify that the uploaded files can be opened and downloaded. **External confirmation**

## 12. Git and final release

- [ ] Review all modified and untracked files.
- [ ] Confirm that no secrets, private credentials, or unnecessary large files are included.
- [ ] Commit the completed pipeline, results, notebook, report, and documentation.
- [ ] Push the final commit to the remote repository.
- [ ] Confirm the public Colab URL resolves to that final commit/branch.
- [ ] Confirm the repository is accessible to the lecturer.
- [ ] Optionally create a final Git tag or GitHub release for the submitted version.

## Final submission gate

Do not mark the project ready until all of the following are complete:

- [x] Report PDF exists and is linked from `README.md`; replace its marked placeholders and rebuild before final submission.
- [ ] Contribution table and percentages are included in the report.
- [ ] Course, lecturer, and member metadata are complete.
- [ ] Team-size issue is resolved.
- [ ] All current work is committed and pushed.
- [ ] A clean public Colab **Run all** succeeds.
- [ ] Administrative registration, progress, deadline, and upload requirements are confirmed.
