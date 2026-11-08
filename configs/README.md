# Cấu hình

`model_experiments.json` là search space được notebook sử dụng trực tiếp:

- seed và số CPU worker;
- tiêu chí chọn `validation_pr_auc`;
- các representation cho từng thuật toán;
- các tổ hợp siêu tham số Logistic Regression và Random Forest.

Hiện có 18 ứng viên Logistic Regression (3 representation × 6 bộ tham số) và 6 ứng viên Random Forest (2 representation × 3 bộ tham số). Chỉnh file JSON rồi chạy lại notebook để tái tạo bảng kết quả và model artifact.

`deep_learning_experiments.json` chứa hai nhóm, mỗi nhóm ba cấu hình:

- `resnet`: embedding dimension, block width/depth, hidden multiplier và dropout;
- `tabm`: các tham số tương tự cùng `member_count` cho số ensemble member nội bộ;
- cả hai: AdamW learning rate/weight decay, batch size, epoch tối đa, patience, positive weight và gradient clipping.

Notebook xếp candidate bằng validation PR-AUC, chọn một winner riêng cho từng thuật toán và khóa threshold bằng validation F1 trước khi mở test.
