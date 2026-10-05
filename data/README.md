# Data

This directory stores the datasets used in the machine learning project.

## Dataset: HR Analytics: Job Change of Data Scientists

- **File:** `aug_train.csv` (1.9 MB)
- **Source:** [Kaggle - HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- **Dimensions:** 19,158 rows × 14 columns
- **Target:** `target` (0 = Candidate not looking for job change, 1 = Candidate looking for job change)
- **Features:**
  - `enrollee_id`: Unique candidate identifier (to be dropped during preprocessing)
  - `city`: City code (123 unique cities; redundant with `city_development_index`)
  - `city_development_index`: Scaled development index of the candidate's city
  - `gender`: Gender of candidate (Male, Female, Other, Missing)
  - `relevent_experience`: Relevant experience of candidate
  - `enrolled_university`: Type of university course enrolled if any
  - `education_level`: Education level of candidate (Primary School, High School, Graduate, Masters, Phd)
  - `major_discipline`: Major discipline of candidate's education
  - `experience`: Candidate total experience in years
  - `company_size`: Number of employees in candidate's current employer
  - `company_type`: Type of current employer
  - `last_new_job`: Difference in years between previous and current job
  - `training_hours`: Training hours completed
