# Tái lập thực nghiệm: AdaMF-MAT và MKG-FENN

Mã nguồn dùng để chạy lại các thực nghiệm trong báo cáo giữa kỳ môn Học máy với dữ liệu đồ thị
(nhóm 25C11005, 25C01015). Kết quả và phân tích nằm trong báo cáo `report/25C11005-25C01015.pdf`.

## Repo này làm gì

Nhóm chạy lại thực nghiệm của hai bài báo về hợp nhất đa mô thức trên đồ thị tri thức:

- **AdaMF-MAT** (LREC-COLING 2024): hoàn thiện đồ thị tri thức đa mô thức (dự đoán liên kết) trên
  ba tập MKG-W, MKG-Y, DB15K. Mã gốc: https://github.com/zjukg/AdaMF-MAT
- **MKG-FENN** (AAAI 2024): dự đoán loại sự kiện tương tác thuốc từ bốn đồ thị tri thức,
  gồm Task 1 (thuốc đã biết) và hai kịch bản cold-start Task 2, Task 3. Mã và dữ liệu kèm bài báo.

## Mã nguồn trong repo

Phần lớn mã là mã gốc của tác giả. Nhóm chỉ sửa ở những chỗ cần để chạy lại được và chạy thêm các
thí nghiệm trong báo cáo:

- AdaMF-MAT
  - Tự chọn thiết bị (CUDA, rồi Apple MPS, rồi CPU) thay cho `.cuda()` cố định.
  - Cờ `-resume`, `-resume_out`, `-total_epoch`: chia một lần chạy dài thành nhiều phiên, nạp lại
    đầy đủ trạng thái mô hình và optimizer.
  - Cờ `-ablation`: bỏ một mô thức (cấu trúc, ảnh hoặc văn bản) khỏi tầng hợp nhất.
  - Cờ `-use_wandb`: ghi log Weights & Biases.
  - `analyze_checkpoint.py` (mới): đọc trọng số mô thức và đo lại khi thiếu mô thức từ một checkpoint.
  - `test_resume.py` (mới): kiểm tra trạng thái huấn luyện (mô hình, optimizer) được giữ nguyên
    qua các phiên.
- MKG-FENN
  - Chạy được trên GPU cho cả ba task (mã gốc chỉ chạy CPU).
  - Sửa `np.mat` thành `np.asmatrix` để chạy với NumPy 2, bỏ `choices` cố định của `--epoches`.
  - Task 1: cờ `--drop_kg` (bỏ một đồ thị tri thức) và `--use_wandb`.

## Cấu trúc

```
AdaMF-MAT/        mã AdaMF-MAT (thư viện mmkgc + script chạy)
  run_adamf_mat.py      huấn luyện và đánh giá link prediction
  args.py               tham số dòng lệnh
  analyze_checkpoint.py trọng số mô thức và thí nghiệm thiếu mô thức từ một checkpoint
  scripts/              siêu tham số gốc cho MKG-W, MKG-Y, DB15K
  test_resume.py        kiểm tra cơ chế chạy tiếp (resume)
MKG-FENN/
  code/                 MKG-FENN-task{1,2,3}.py, modeltask{1,2,3}.py
  result/               nơi mã ghi all/each task*.csv
results/                file CSV xuất từ các run W&B và số liệu thiếu mô thức mà báo cáo sử dụng
report/                 mã LaTeX và bản PDF của báo cáo
requirements.txt
```

## Môi trường và phần cứng

- GPU NVIDIA Tesla T4 16GB, CUDA 12.8. Mã cũng chạy được trên CPU và Apple MPS, nhưng chậm hơn nhiều.
- Python 3.12, PyTorch 2.10.0. Cài thư viện:

```bash
pip install -r requirements.txt
```

- Trình biên dịch C++ (g++ hoặc clang) để build thư viện lấy mẫu của AdaMF-MAT.

Thời gian tham khảo trên một GPU T4: AdaMF-MAT MKG-W 1000 epoch khoảng 4 giờ, MKG-Y khoảng 2 giờ,
DB15K khoảng 32 giờ; MKG-FENN mỗi lần chạy (5-fold, 100 đến 120 epoch) khoảng 1 giờ.

## Dữ liệu

Dữ liệu không kèm trong repo, lấy từ nguồn gốc:

- AdaMF-MAT: đặt `benchmarks/` (MKG-W, MKG-Y, DB15K) từ repo gốc vào `AdaMF-MAT/benchmarks/`, và
  embeddings ảnh/văn bản (`<DATASET>-visual.pth`, `<DATASET>-textual.pth`) tải từ
  https://drive.google.com/drive/folders/1UJSfnb8DEx2s-k8zaQx1fWUw5f45GBpI vào `AdaMF-MAT/embeddings/`.
- MKG-FENN: tạo thư mục `MKG-FENN/dataset/` chứa `dataset1.txt` đến `dataset4.txt` và `event.db`
  (giải nén từ `event.rar` trong mã gốc), ngang hàng với `code/` và `result/`.

## AdaMF-MAT

Chạy trong thư mục `AdaMF-MAT/`. Build thư viện lấy mẫu một lần cho mỗi máy:

```bash
cd mmkgc && bash make.sh && cd ..
```

Hệ số MAT truyền qua `-mu` (bài báo ký hiệu là λ; `-mu=0` là tắt MAT). Siêu tham số chung theo
`scripts/run_*.sh`: `-batch_size=1024 -neg_num=128 -learning_rate=2e-5 -epoch=1000`; MKG-W và DB15K
dùng `-margin=12`, MKG-Y dùng `-margin=4`; MKG-W và MKG-Y dùng `-dim=200`, DB15K dùng `-dim=250`.

```bash
# MKG-W, lưới mu và nhiều seed (đổi -mu thuộc {0, 0.001, 0.01, 0.1}, -seed thuộc {42, 1, 2})
python run_adamf_mat.py -dataset=MKG-W -batch_size=1024 -margin=12 -epoch=1000 -dim=200 \
  -neg_num=128 -learning_rate=2e-5 -mu=0 -seed=42 -save=checkpoint/mkgw_mu0

# MKG-Y, lưới mu
python run_adamf_mat.py -dataset=MKG-Y -batch_size=1024 -margin=4 -epoch=1000 -dim=200 \
  -neg_num=128 -learning_rate=2e-5 -mu=0.01 -save=checkpoint/mkgy_mu001

# Ablation mô thức trên MKG-W (-ablation thuộc {no_s, no_v, no_t}: bỏ cấu trúc, ảnh, văn bản)
python run_adamf_mat.py -dataset=MKG-W -batch_size=1024 -margin=12 -epoch=1000 -dim=200 \
  -neg_num=128 -learning_rate=2e-5 -mu=0 -ablation=no_v -save=checkpoint/mkgw_abl_nov
```

DB15K chạy 1000 epoch theo bốn phiên 250 epoch. Phiên đầu không có `-resume`; các phiên sau truyền
file trạng thái của phiên trước. Chỉ phiên cuối (đạt `-total_epoch`) mới đánh giá trên tập test.

```bash
python run_adamf_mat.py -dataset=DB15K -batch_size=1024 -margin=12 -dim=250 -neg_num=128 \
  -learning_rate=2e-5 -mu=0 -epoch=250 -total_epoch=1000 \
  -resume_out=resume_db15k_mu0.pt -save=checkpoint/db15k_mu0
# lặp lại 3 lần
python run_adamf_mat.py -dataset=DB15K -batch_size=1024 -margin=12 -dim=250 -neg_num=128 \
  -learning_rate=2e-5 -mu=0 -epoch=250 -total_epoch=1000 \
  -resume=resume_db15k_mu0.pt -resume_out=resume_db15k_mu0.pt -save=checkpoint/db15k_mu0
```

Trọng số mô thức và thí nghiệm thiếu mô thức, từ checkpoint MKG-W `mu=0`:

```bash
python analyze_checkpoint.py --dataset MKG-W --ckpt checkpoint/mkgw_mu0 --mode attention
python analyze_checkpoint.py --dataset MKG-W --ckpt checkpoint/mkgw_mu0 --mode missing --out fig3_missing.csv
```

## MKG-FENN

Chạy trong thư mục `MKG-FENN/code/`; kết quả ghi vào `../result/`.

```bash
# Task 1, ba seed
python MKG-FENN-task1.py --epoches 120 --seed 0      # tương tự --seed 1, --seed 2

# Bỏ một đồ thị tri thức (--drop_kg thuộc {1, 2, 3, 4}), mỗi cấu hình 3 seed {0, 1, 2}
python MKG-FENN-task1.py --epoches 120 --drop_kg 3 --seed 0

# Độ nhạy siêu tham số, mỗi giá trị 3 seed {0, 1, 2}
python MKG-FENN-task1.py --epoches 120 --neighbor_sample_size 10 --seed 0   # thuộc {4, 6, 10, 16}
python MKG-FENN-task1.py --epoches 120 --embedding_num 256 --seed 0         # thuộc {32, 64, 128, 256}

# Cold-start: Task 2 (một thuốc mới) và Task 3 (hai thuốc mới), mỗi task 3 seed {1, 2, 3}
python MKG-FENN-task2.py --epoches 100 --seed 1   # mặc định của mã; lặp với --seed 2, 3
python MKG-FENN-task3.py --epoches 120 --seed 1
```

## Ghi log Weights & Biases

`run_adamf_mat.py` và `MKG-FENN-task1.py` có cờ ghi log W&B, mặc định tắt. Đăng nhập bằng
`wandb login`, sau đó thêm cờ:

```bash
python run_adamf_mat.py ... -use_wandb=1 -wandb_project=<project> -wandb_name=<tên run>
python MKG-FENN-task1.py ... --use_wandb 1 --wandb_project <project> --wandb_name <tên run>
```

Cờ tùy chọn `wandb_entity`, `wandb_group`, `wandb_notes` (cùng tiền tố `-` hoặc `--` như trên) đặt
entity, nhóm run và ghi chú. Mỗi run ghi loss và chỉ số theo epoch trong lúc huấn luyện; MKG-FENN
ghi thêm epoch được chọn và macro-F1 ở epoch cuối của từng fold. Thông tin máy chạy (tên máy, đường
dẫn, danh sách thư viện) không được gửi lên W&B.

## Biên dịch báo cáo

Cần TeX Live (hoặc TinyTeX) có các gói `IEEEtran`, `vntex`, `babel-vietnamese`, `tikz`.

```bash
cd report
pdflatex main && pdflatex main
```
