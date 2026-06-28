# Ke Hoach Tong The Tai Tao Mini Move-in-2D

## 0. Muc tieu

Du an nay nham tai tao bai toan **Move-in-2D: 2D-Conditioned Human Motion Generation** o quy mo nho hon, theo huong nghien cuu co kiem chung.

Bai toan can tai tao:

> Cho mot anh canh 2D va mot mo ta hanh dong bang text, sinh ra chuyen dong nguoi phu hop voi ca hanh dong va boi canh trong anh.

Ban mini khong nham dat chat luong nhu paper goc. Muc tieu chinh la xay duoc mot pipeline day du de chung minh logic:

```text
2D scene image + text prompt -> human motion -> visualization/inference -> evaluation
```

## 1. Xay lai dataset

Muc tieu cua buoc nay la tao mot dataset nho, sach, co format thong nhat va co the dung de train/test.

Dataset can gom cac mau co dang:

```text
scene image + text description + human motion
```

Trong do:

- `scene image`: anh canh 2D lam dieu kien boi canh.
- `text description`: mo ta hanh dong cua nguoi, vi du `a person walks`, `a person sits`, `a person dances`.
- `human motion`: chuoi chuyen dong nguoi, co the bat dau bang 2D keypoints thay vi SMPL de giam do phuc tap.

Pham vi ban dau:

- Khong can dataset lon nhu HiC-Motion.
- Chi can du so luong de chung minh pipeline hoc duoc quan he giua scene, text va motion.
- Nen bao phu mot so nhom hanh dong ro rang: di, chay, ngoi, nhay, mua.

Dau ra mong muon cua buoc nay:

- Mot dataset nho co cau truc ro rang.
- Co metadata mo ta moi sample.
- Co train/test split.
- Co script hoac quy trinh kiem tra nhanh chat luong du lieu.

## 2. Dung model

Muc tieu cua buoc nay la dung mot phien ban nho cua conditional human motion generation model.

Model can nhan vao:

- Anh canh 2D.
- Text prompt.

Model can sinh ra:

- Chuoi motion nguoi.

Representation ban dau nen don gian:

- Uu tien 2D keypoints hoac motion vector don gian.
- Chua can dung SMPL neu muc tieu la ban mini de kiem chung y tuong.

Tinh than can giu tu paper goc:

- Motion khong chi phu thuoc vao text.
- Motion phai duoc dieu kien hoa boi anh canh 2D.
- Co the so sanh model day du voi baseline chi dung text de thay vai tro cua scene.

Dau ra mong muon cua buoc nay:

- Mot model nho co the train duoc tren dataset mini.
- Interface ro rang cho training va inference.
- Co baseline don gian de so sanh.

## 3. Train thu lai

Muc tieu cua buoc nay la kiem tra xem pipeline co hoc duoc tin hieu hay khong, khong phai dat ket qua SOTA.

Huong train ban dau:

- Dung cau hinh nho, de debug.
- Train trong thoi gian ngan.
- Theo doi loss va mot vai chi so truc quan.
- Kiem tra dinh ky bang visualization.

Can quan sat:

- Model co hoc duoc action theo text khong.
- Model co su dung thong tin scene khong.
- Motion co on dinh va nam trong khung hinh khong.
- Motion co qua nhieu jitter, drift hoac collapse khong.

Dau ra mong muon cua buoc nay:

- Checkpoint model da train.
- Log train co the doc lai.
- Mot vai sample inference trong qua trinh train.
- Nhan xet ban dau ve kha nang hoc cua model.

## 4. Infer va test

Muc tieu cua buoc nay la tao pipeline inference de tu anh va prompt sinh ra motion co the quan sat duoc.

Input inference:

```text
scene image + text prompt
```

Output inference:

```text
generated motion + visualization
```

Can test tren:

- Prompt da co trong tap train.
- Prompt gan nghia nhung khac cau chu.
- Anh canh moi.
- Cac action khac nhau.
- Truong hop kho hoac khong hop le.

Visualization la bat buoc vi day la cach nhanh nhat de danh gia ban mini:

- Ve skeleton len anh.
- Xuat video ngan.
- Kiem tra trajectory, scale, vi tri va do tu nhien cua motion.

Neu muon mo rong sang video nguoi that:

- Day nen la buoc noi them sau khi motion generation on dinh.
- Khong nen train video generation model tu dau.
- Co the dung model co san de bien pose/motion thanh video.

Dau ra mong muon cua buoc nay:

- Script inference chay duoc.
- Mot tap demo output.
- Video/animation de xem truc quan.
- Ghi chu cac case thanh cong va that bai.

## 5. So sanh va danh gia

Muc tieu cua buoc nay la danh gia xem ban mini tai tao duoc den dau logic cua Move-in-2D.

Can co baseline:

- Baseline chi dung text, khong dung scene.
- Baseline retrieval hoac rule-based neu can mot moc so sanh don gian.
- Model day du dung ca text va scene.

Cac truc danh gia:

- **Text alignment**: motion co dung hanh dong trong prompt khong.
- **Scene alignment**: motion co phu hop voi boi canh anh khong.
- **Motion quality**: motion co lien tuc, it jitter, it loi hinh hoc khong.
- **Diversity**: voi cung loai prompt, model co sinh duoc nhieu motion hop ly khong.
- **Robustness**: doi anh hoac doi prompt thi output co on dinh khong.

Dau ra mong muon cua buoc nay:

- Bang so sanh giua cac baseline va model day du.
- Tap visualization minh hoa thanh cong/that bai.
- Nhan xet ro model hoc duoc gi va chua hoc duoc gi.
- Ket luan ve gioi han cua ban mini.

## 6. Thu tu trien khai de xuat

Nen lam theo thu tu:

1. Chot format dataset va representation cua motion.
2. Xay dataset mini.
3. Dung baseline don gian.
4. Dung model conditional nho.
5. Train thu va visualize lien tuc.
6. Viet inference pipeline.
7. Danh gia va so sanh voi baseline.
8. Neu can, moi mo rong sang video nguoi that hoac SMPL.

## 7. Nguyen tac pham vi

De giu du an kha thi, ban mini nen mac dinh:

- Uu tien 2D keypoints truoc SMPL.
- Uu tien skeleton visualization truoc video nguoi that.
- Uu tien dataset nho, sach truoc dataset lon nhung nhieu noise.
- Uu tien pipeline day du truoc model phuc tap.
- Uu tien so sanh co baseline truoc khi toi uu chat luong.

## 8. Cac plan chi tiet can viet tiep

Moi buoc lon nen co mot plan rieng truoc khi implement:

- `DATASET_PLAN.md`: cach tao dataset, format, preprocessing, split, quality control.
- `MODEL_PLAN.md`: architecture, input/output, conditioning, baseline.
- `TRAINING_PLAN.md`: cau hinh train, loss, checkpoint, logging.
- `INFERENCE_PLAN.md`: CLI/API inference, visualization, output format.
- `EVALUATION_PLAN.md`: metric, baseline, protocol so sanh, report format.

