# Paper-like Data Pipeline

## 0. Muc tieu

Tai lieu nay mo ta nhanh xu ly du lieu theo huong gan voi paper **Move-in-2D** hon.

Hien project dang co nhanh du lieu dau tien:

```text
GT branch: Penn Action annotation goc -> mini_move_in_2d
```

Nhanh nay dung nhan co san cua Penn Action:

- action label goc,
- 2D keypoints goc,
- frame goc lam scene image.

Nhanh moi can them:

```text
Paper-like branch: raw frames -> off-the-shelf models -> pseudo labels/background -> paper_like_mini_move_in_2d
```

Y tuong la khong dung truc tiep nhan keypoint goc de tao motion nua, ma dung cac model co san de tu xu ly video/frame giong tinh than paper. Sau do so sanh pseudo labels voi nhan Penn Action goc de xem pipeline tu dong co dang tin khong.

### Quyet dinh dataset v1

Sau cac test nho, nhanh background v1 chot dung:

```text
Penn Action bbox/keypoint window -> padded bbox mask -> LaMa inpainting -> human-removed scene image
```

Ly do: Mask R-CNN mask qua sat nguoi nen LaMa co xu huong ve lai nguoi/nguoi nem bong. Bbox mask rong hon xoa ca vung nguoi va vat the lien quan, cho background sach hon trong cac sample da test.

Dataset v1 se giu:

- motion: Penn Action 2D keypoints da xu ly san,
- text: prompt tu action label,
- scene image: bbox + LaMa background.

Pseudo pose/SMPL theo paper se de sang nhanh V2 sau khi dataset v1 train/infer duoc.

## 1. Tong quan pipeline

Pipeline du kien:

```text
Raw Penn Action frames
-> Check anh/frame quality
-> Detect nguoi va loc single-person
-> Extract pose/motion pseudo-label
-> Segment/mask nguoi
-> Inpaint/remove nguoi khoi scene frame
-> Build paper-like dataset
-> Compare GT vs pseudo
```

Ket qua cuoi cung can co:

```text
project_data/datasets/penn_action/paper_like_processed/
project_data/datasets/penn_action/paper_like_mini_move_in_2d/
project_data/datasets/penn_action/comparison_gt_vs_pseudo/
```

## 2. Buoc 1: Check anh / frame quality

### Muc dich

Dam bao raw frames doc duoc va co chat luong toi thieu truoc khi dua vao cac model detection, pose, segmentation.

### Tool

- `Pillow`
- `OpenCV`

### Input

```text
project_data/datasets/penn_action/raw/Penn_Action/frames/<video_id>/*.jpg
```

### Output

```text
project_data/datasets/penn_action/paper_like_processed/frame_quality/frame_quality_manifest.jsonl
```

Moi row nen gom:

```text
video_id
frame_path
frame_index
readable
width
height
brightness_mean
blur_score
quality_flags
```

### Logic co ban

- Doc tung frame.
- Kiem tra frame co bi loi/corrupt khong.
- Lay kich thuoc anh.
- Tinh brightness trung binh.
- Tinh blur score bang variance of Laplacian.
- Danh dau frame qua toi, qua sang, qua mo, hoac khong doc duoc.
- Tong hop len video-level de biet video nao co qua nhieu frame loi.

### Ket qua sau buoc nay

Ta co danh sach frame hop le va cac video co the dua vao cac model phia sau.

## 3. Buoc 2: Detect nguoi / single-person filtering

### Muc dich

Gan voi paper: tac gia loc video co mot nguoi chinh trong scene bang keypoint-based/person detection models.

### Tool trong paper

- `Keypoint R-CNN` cho person detection.
- `OpenPose` cho keypoint prediction.

### Tool practical co the dung

- `torchvision` Keypoint R-CNN.
- `Detectron2` Keypoint R-CNN.
- `YOLO pose`.
- Neu muon setup nhe hon, co the dung mot pose/detection model de lam ca detection va keypoint.

### Input

```text
frame_quality_manifest.jsonl
raw frames
```

### Output

```text
project_data/datasets/penn_action/paper_like_processed/person_detection/<video_id>.json
```

Moi frame nen gom:

```text
video_id
frame_index
detections: [
  {
    bbox_xyxy,
    score,
    class_name
  }
]
primary_person
num_persons
detection_flags
```

### Logic co ban

- Chay detector tren tung frame.
- Lay cac detection class `person`.
- Chon primary person:
  - uu tien bbox lon nhat,
  - uu tien detection score cao,
  - neu co tracking thi uu tien track on dinh qua frame.
- Danh dau:
  - frame khong co nguoi,
  - frame co nhieu nguoi,
  - bbox qua nho,
  - detection nhay bat thuong.
- O video-level, tinh ti le frame co dung mot nguoi chinh.

### Ket qua sau buoc nay

Ta co detection cua nguoi chinh tren tung frame va co the loc nhung video/frame khong hop voi bai toan single-human motion.

## 4. Buoc 3: Extract pose / motion pseudo-label

### Muc dich

Tu raw frames, dung model co san de tao motion annotation tu dong. Day la phan gan voi paper nhat ve mat pseudo-labeling.

### Tool trong paper

- `OpenPose` de predict 2D keypoints.
- `4D-Humans` de extract pseudo ground-truth motion o dang `SMPL`.

### Tool practical phase dau

- Bat dau bang pseudo 2D pose de so sanh truc tiep voi Penn Action GT 2D.
- Sau khi pipeline on dinh moi nang len `4D-Humans -> SMPL`.

### Input

```text
raw frames
person_detection/<video_id>.json
```

### Output

```text
project_data/datasets/penn_action/paper_like_processed/pseudo_pose_raw/<video_id>.npz
```

Noi dung output:

```text
keypoints_2d_px
keypoints_2d_norm
keypoint_scores
visibility
bbox_xyxy
root_xy
frame_indices
detection_scores
```

### Logic co ban

- Chay pose estimator tren tung frame.
- Neu model tra nhieu nguoi, chon pose gan voi `primary_person` bbox.
- Chuan hoa keypoints ve:
  - pixel coordinate,
  - normalized coordinate theo width/height.
- Tinh root trajectory.
- Tinh missing joints va confidence.
- Chua filter qua manh o buoc nay; luu raw pseudo output truoc de audit.

### Ket qua sau buoc nay

Ta co pseudo motion/frame-level pose duoc extract tu model thay vi dung keypoint label goc.

## 5. Buoc 4: Mask nguoi

### Muc dich

Tao person mask de remove nguoi khoi scene frame, giong paper dung Mask R-CNN de detect person mask.

### Tool trong paper

- `Mask R-CNN`.

### Tool practical co the dung

- `torchvision` Mask R-CNN.
- `Detectron2` Mask R-CNN.
- `SAM`/`SAM2` neu can mask chat luong cao.
- Co the bat dau voi segmentation model nhe hon neu setup Mask R-CNN kho.

### Input

```text
scene frame candidates
person_detection/<video_id>.json
```

### Output

```text
project_data/datasets/penn_action/paper_like_processed/person_masks/<sample_id>.png
```

Moi mask nen la anh grayscale/binary:

```text
0 = background
255 = person region
```

### Logic co ban

- Chon scene frame cho moi sample/window.
- Segment nguoi trong scene frame.
- Neu co nhieu mask, chon mask overlap voi primary bbox.
- Dilate mask nhe de che het vien co the.
- Luu mask va metadata:
  - mask area,
  - bbox,
  - segmentation confidence,
  - failure flags.

### Ket qua sau buoc nay

Ta co mask cua nguoi trong scene frame, san sang cho inpainting.

## 6. Buoc 5: Inpaint / remove nguoi khoi background

### Muc dich

Tao background image sach de model condition vao scene, khong nhin thay nguoi goc trong frame.

### Tool trong paper

- Mot basic inpainting model.

### Tool practical co the dung

- Baseline nhe: `OpenCV inpaint`.
- Ban tot hon: `LaMa`.
- Ban nang hon: diffusion-based inpainting.

### Input

```text
original scene frame
person mask
```

### Output

```text
project_data/datasets/penn_action/paper_like_processed/backgrounds_inpainted/<sample_id>.jpg
project_data/datasets/penn_action/paper_like_processed/background_previews/<sample_id>.jpg
```

### Logic co ban

- Doc original scene frame.
- Doc person mask.
- Chay inpainting de lap vung nguoi bang background xung quanh.
- Luu anh background da remove nguoi.
- Tao preview before/after de kiem tra truc quan.

### Ket qua sau buoc nay

Ta co background image gan voi paper hon frame goc, dung lam scene condition cho model motion generation.

## 7. Buoc 6: Build paper-like dataset

### Muc dich

Gom pseudo motion va inpainted background thanh dataset co interface giong GT branch hien tai.

### Tool

- Script noi bo cua project.

### Input

```text
pseudo_pose_raw/<video_id>.npz
backgrounds_inpainted/<sample_id>.jpg
action label/text prompt tu Penn Action
frame_quality_manifest.jsonl
person_detection metadata
```

### Output

```text
project_data/datasets/penn_action/paper_like_mini_move_in_2d/manifest_filtered.jsonl
project_data/datasets/penn_action/paper_like_mini_move_in_2d/manifest_rejected.jsonl
project_data/datasets/penn_action/paper_like_mini_move_in_2d/splits.json
project_data/datasets/penn_action/paper_like_mini_move_in_2d/dataset_report.md
```

Motion files:

```text
project_data/datasets/penn_action/paper_like_processed/motions/*.npz
```

Background files:

```text
project_data/datasets/penn_action/paper_like_processed/scenes_inpainted/*.jpg
```

### Logic co ban

- Cat pseudo pose thanh window 64 frames.
- Match frame_start/frame_end voi GT branch neu co the.
- Tinh quality signals:
  - missing keypoint ratio,
  - bbox size,
  - out-of-frame ratio,
  - root displacement,
  - temporal jitter,
  - detection stability.
- Filter sample xau.
- Split theo `video_id`, khong split theo frame/window.
- Luu manifest co schema gan giong GT branch.

### Ket qua sau buoc nay

Ta co mot dataset paper-like rieng, co the dung de train hoac so sanh voi GT branch.

## 8. Buoc 7: So sanh GT vs pseudo

### Muc dich

Kiem tra pipeline tu dong co tao pseudo labels du tot khong.

### Tool

- Script noi bo cua project.
- Visualization bang `OpenCV`, `Pillow`, hoac `matplotlib`.

### Input

GT branch:

```text
project_data/datasets/penn_action/mini_move_in_2d/manifest_filtered.jsonl
```

Paper-like branch:

```text
project_data/datasets/penn_action/paper_like_mini_move_in_2d/manifest_filtered.jsonl
```

### Output

```text
project_data/datasets/penn_action/comparison_gt_vs_pseudo/report.md
project_data/datasets/penn_action/comparison_gt_vs_pseudo/metrics.json
project_data/datasets/penn_action/comparison_gt_vs_pseudo/previews/*.jpg
```

### Metrics

- Keypoint L2 error.
- PCK.
- Bbox IoU.
- Missing joint rate.
- Root trajectory error.
- Temporal jitter.
- Filter agreement/disagreement.
- Per-action quality breakdown.

### Logic co ban

- Match GT sample va pseudo sample theo:
  - `video_id`,
  - `frame_start`,
  - `frame_end`.
- Load GT motion va pseudo motion.
- Dua ve cung joint format neu can.
- Tinh metric theo sample/action/split.
- Render preview:
  - GT skeleton mot mau,
  - pseudo skeleton mot mau,
  - bbox GT va pseudo,
  - original frame va inpainted background.

### Ket qua sau buoc nay

Ta co cau tra loi ro:

```text
Pseudo-label tu off-the-shelf models co du tot de train mini Move-in-2D khong?
```

## 9. Giai thich tool chi tiet

### OpenCV

Dung cho:

- doc/ghi frame,
- tinh blur score,
- tinh brightness,
- resize,
- ve preview,
- inpainting baseline.

Uu diem:

- nhe,
- de cai,
- chay CPU duoc.

Nhuoc diem:

- inpainting OpenCV chi la baseline, de bi artifact voi mask lon.

### Pillow

Dung cho:

- doc/ghi anh,
- tao preview don gian,
- convert format anh.

Uu diem:

- on dinh,
- nhe,
- phu hop voi thao tac image co ban.

### Keypoint R-CNN

Trong paper, Keypoint R-CNN duoc dung cho person detection/filtering. Ban practical co the dung implementation trong `torchvision` hoac `Detectron2`.

Dung cho:

- detect person,
- detect keypoints neu dung ban keypoint.

Uu diem:

- gan voi paper,
- model co san.

Nhuoc diem:

- can PyTorch/torchvision,
- chay full dataset co the cham neu CPU-only.

### OpenPose

Trong paper, OpenPose duoc dung de predict keypoints.

Dung cho:

- extract 2D keypoints tu frame.

Uu diem:

- dung tool gan voi paper.

Nhuoc diem:

- setup co the phuc tap,
- dependency cu,
- co the kho tich hop hon cac pose estimator moi.

### Mask R-CNN

Trong paper, Mask R-CNN duoc dung de detect person masks.

Dung cho:

- tao binary mask cua nguoi,
- phuc vu inpainting background.

Uu diem:

- dung voi tinh than paper,
- mask instance-level.

Nhuoc diem:

- mask co the thieu chi tiet o bien nguoi,
- can model weights va inference runtime.

### 4D-Humans

Trong paper, 4D-Humans duoc dung de extract pseudo ground-truth motion dang `SMPL`.

Dung cho:

- tao SMPL pose sequence,
- motion 3D/parametric gan voi paper hon 2D keypoints.

Uu diem:

- sat paper nhat ve motion representation.

Nhuoc diem:

- nang,
- can GPU de thuc te,
- output kho debug hon 2D,
- Penn Action khong co GT SMPL de so sanh truc tiep.

### Inpainting model

Dung cho:

- remove nguoi khoi scene frame,
- tao background image sach.

Lua chon:

- `OpenCV inpaint`: nhanh, nhe, baseline.
- `LaMa`: chat luong tot hon, can setup model.
- Diffusion inpainting: co the dep hon nhung nang va cham hon.

### Comparison scripts

Dung cho:

- so sanh pseudo labels voi GT Penn Action,
- tinh metric,
- render preview thanh cong/that bai.

Day la phan quan trong vi Penn Action cho phep danh gia pipeline tu dong, dieu ma paper tren internal videos khong the hien chi tiet.

## 10. Thu tu nen lam

Thu tu de xuat:

1. Viet frame quality checker.
2. Chay thu detector/pose estimator tren 5-10 video.
3. Chot tool pose/detection practical.
4. Tao pseudo pose raw outputs.
5. Viet comparison GT vs pseudo cho pose.
6. Sau khi pose on, them mask + inpainting.
7. Build paper-like dataset.
8. Tao report tong hop.
9. Neu can sat paper hon, moi them 4D-Humans/SMPL.

## 10.1. Lenh da co trong repo

Script dau tien cho nhanh paper-like:

```bash
python3 scripts/penn_action/paper_like_pipeline.py --help
```

Kiem tra layout va tool hien co:

```bash
python3 scripts/penn_action/paper_like_pipeline.py ensure-layout
python3 scripts/penn_action/paper_like_pipeline.py status
python3 scripts/penn_action/paper_like_pipeline.py tool-check
```

Chay buoc 1 tren mot subset nho:

```bash
python3 scripts/penn_action/paper_like_pipeline.py frame-quality --max-videos 10
```

Chay buoc 1 tren toan bo Penn Action:

```bash
python3 scripts/penn_action/paper_like_pipeline.py frame-quality
```

Tao baseline inpainting bang GT bbox de test wiring mask -> inpaint -> preview:

```bash
python3 scripts/penn_action/paper_like_pipeline.py gt-mask-inpaint-baseline --max-samples 32
```

Mac dinh moi lan chay se tao mot folder run rieng:

```text
project_data/datasets/penn_action/paper_like_processed/experiments/gt_mask_inpaint_baseline/<run_id>/
  config.json
  manifest.jsonl
  person_masks/
  backgrounds_inpainted/
  background_previews/
```

Co the dat ten run de tien so sanh:

```bash
python3 scripts/penn_action/paper_like_pipeline.py gt-mask-inpaint-baseline --max-samples 5 --run-name bbox_cv2_telea_test_001
```

Luu y: baseline tren dung GT bbox, nen khong phai paper-like segmentation cuoi cung. No chi de kiem tra ha tang output/inpainting truoc khi gan Mask R-CNN/SAM. Cac ket qua cu o `person_masks/`, `backgrounds_inpainted/`, `background_previews/` duoc giu lai de review, khong xoa.

Build dataset v1 bang bbox + LaMa:

```bash
python3 scripts/penn_action/paper_like_pipeline.py build-bbox-lama-dataset --max-samples 50 --run-name bbox_lama_dataset_50_test_001 --mask-pad 32 --device cpu
```

Output moi lan chay nam trong folder rieng:

```text
project_data/datasets/penn_action/paper_like_mini_move_in_2d/<run_id>/
  config.json
  manifest_filtered.jsonl
  splits.json
  dataset_report.md
  failures.jsonl
  scenes/
  bbox_masks/
  previews/
```

## 11. Ghi chu ve pham vi

- Khong overwrite GT branch hien tai.
- Paper-like branch phai luu rieng.
- Moi lan thu nghiem nen ghi vao folder con co `run_id`, kem `config.json` va `manifest.jsonl`.
- Khong xoa output cu; cac output debug/baseline duoc giu de so sanh va lam bao cao.
- Phase dau co the dung pseudo 2D pose de kiem chung pipeline.
- Phase sau moi nang len SMPL/4D-Humans neu can.
- Tat ca model weights/cache phai nam trong:

```text
project_data/models/
```
