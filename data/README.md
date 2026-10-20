# Dữ liệu

## HR Analytics: Job Change of Data Scientists

- File: `aug_train.csv` (khoảng 1,9 MB).
- Nguồn: [Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists).
- Kích thước: 19.158 dòng × 14 cột.
- Target: `target` (`0`: không muốn chuyển việc, `1`: muốn chuyển việc).

Nhóm cột:

- Định danh: `enrollee_id`.
- Địa lý: `city`, `city_development_index`.
- Cá nhân/học vấn: `gender`, `enrolled_university`, `education_level`, `major_discipline`.
- Kinh nghiệm/công việc: `relevent_experience`, `experience`, `company_size`, `company_type`, `last_new_job`.
- Đào tạo: `training_hours`.

File dữ liệu thô được giữ nguyên. Mọi thao tác làm sạch và biến đổi nằm trong notebook/module để bảo đảm khả năng tái lập.
