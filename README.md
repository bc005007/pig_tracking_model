# pig_tracking_model
YOLOv5 & StrongSORT


### 개요

- YOLOv5 + STRONGSORT를 활용한 양돈 객체 검출 및 TRACKING

### Training Environment

- 각 모델 설정값 참고
- KiSTi 국가컴퓨팅센터 - 뉴론(NEURON)
- 운영체제: CentOS 7.9 (Linux, 64-bit)
- CPU: Intel Xeon Ivy Bridge (E5-2670) / 2.50GHz (10-core) / 2 socket
- GPU: Tesla V100-PCIE-16GB(2개중 하나만 사용함)
- 메인 메모리: 128GB DDR3 Memory
- CUDA Version: 11.6
- Python Version: 3.9.13
- YOLO Ultraylitics

### Yolov5-STRONG SORT모델

```python
Yolov5 StrongSort모델

┌─── data                 
│    ├── annotation   ├── train.json                 
│                     ├── val.json 
│                     ├── test.json              
│    ├── image                
├─── pig                  
│    ├── images ├── train
│               ├── test
│								├── val
│    ├── labels
│    ├── custom_data.yaml
├─── anaconda3
│    ├── 
├─── COCO2YOLO
│    ├── test.py
│    ├── COCO2YOLOY.py
│    ├── README.md
├─── ultra_workdir
│    ├── best.pt
│    ├── last.pt
├─── yolov5
│    ├── train.py
│    ├── val.py
│    ├── detect.py
├─── yolov5_new
│    ├── COCO2YOLO 
│    ├── DeepSORT
│               ├── yolov5_STRONGSORT
│                         ├── trackers  
│                         ├── track2.py  
```

### 2. Model

2.1 모델- Yolov5 Ultralytics

1.Ultralytics Package 설치 및 requirements 설치

```python
!git clone https://github.com/ultralytics/yolov5
!cd yolov5;pip install -qr requirements.txt
```

2.split_dataset- 1가지 COCO dataset을  train, val, test 셋으로 나눈다(val_ratio=0.2, test_ratio=0.1, random_seed=221111)

```python
split_dataset(input_json='data/annotation/annotation.json',
              output_dir='data/annotation',
              val_ratio=0.2,
              test_ratio=0.1,
              random_seed=221111)
```

3.COCO2YOLO COCO 데이터 셋을 YOLO 형태로 변환

```python
# train 용 yolo 데이터 세트 생성. 
train_yolo_converter = COCO2YOLO(src_img_dir='/data/data/image', json_file='data/annotation/seed13/train.json',
                                 tgt_img_dir='/data/pig/images/train', tgt_anno_dir='/data/pig/labels/train')
train_yolo_converter.coco2yolo()

# val 용 yolo 데이터 세트 생성. 
val_yolo_converter = COCO2YOLO(src_img_dir='/data/data/image', json_file='data/annotation/seed13/train.json',
                                 tgt_img_dir='/data/pig/images/val', tgt_anno_dir='/data/pig/labels/val')
val_yolo_converter.coco2yolo()

# test 용 yolo 데이터 세트 생성. 
test_yolo_converter = COCO2YOLO(src_img_dir='/data/data/image', json_file='data/annotation/seed13/train.json',
                                 tgt_img_dir='/data/pig/images/test', tgt_anno_dir='/data/pig/labels/test')
test_yolo_converter.coco2yolo()
```

4.모델 학습

custom_data.yaml 파일 학습

```python
!cd /data/yolov5; python train.py --img 640 --batch 8 --epochs 50 --data pig/custom_data.yaml --weights yolov5l.pt \
                                     --project=ultra_workdir--name pig --exist-ok
```

학습된 모델 inference 돌리기(잘 돌아가는 지 확인)

```python
!cd /content/yolov5;python detect.py --source /content/pig/images/test/ \
                            --weights /mydrive/ultra_workdir/pig/weights/epoch50_batchsize8.pt --conf 0.2 \
                            --project=/content/data/output --name=run_image --exist-ok --line-thickness 2
```

2.2 모델- STRONG SORT (yolov5에서 구한 weight과 STRONG SORT의 가중치를 넣어서 STRONG SORT 실행함)

```python
!python track.py --yolo-weights epoch50_batchsize8.pt --reid-weights osnet_x0_25_msmt17.pt --source /data/yolov5_new/pigvid1.mp4 --line-thickness 2 --conf-thres 0.8 --iou-thres 0.5 --augment --save-vid --save-txt
```

track.py

yolo 가중치(학습한 모델: epoch50_batchsize8.pt

STRONG SORT의 가중치: osnet_x0_25_msmt17.pt

--source(영상): /data/yolov5_new/pigvid1.mp4

### 3. How to use

**YOLOV5-ultralytics 패키지**

3-1.  custom_data.yaml파일을 yolo로 돌림

```python
!cd /data/yolov5; python train.py --img 640 --batch 8 --epochs 30 --data pig/custom_data.yaml --weight yolov51.pt \
                                     --project=ultra_workdir--name pig --exist-ok
```

3-2. INFERENCE

학습된 모델이 잘 학습 되었는지 확인하기 위해 {확인하고 싶은 사진}을 INFERENCE 시킴

```python
!cd /data/yolov5_new/yolov5;python detect.py --source {확인하고 싶은 사진} \
                            --weights /data/yolov5_new/ultra_workdir/epoch50/pig/weights/best.pt --conf 0.2 \
                            --project=/data/yolov5_new/output --name=run_image --exist-ok --line-thickness 2
```

**STRONG SORT**

3-3. YOLO를 돌려서 나온 .pt 파일}을 STRONG SORT에 넣어서 Tracking 시작

```python
!python track_2.py --yolo-weights {YOLO를 돌려서 나온 .pt 파일} --reid-weights osnet_x0_25_msmt17.pt --source /data/yolov5_new/pig_sample.mp4 --line-thickness 2 --conf-thres 0.8 --iou-thres 0.5 --augment --save-vid --save-txt
```

{YOLO를 돌려서 나온 .pt}- yolo를 돌린 후 /mydrive/ultra_workdir/pig/weights/에서 나온 {   .pt파일} 

구한 결과 값: 

track.py

yolo 가중치: epoch50_batchsize8.pt

STRONG SORT의 가중치: osnet_x0_25_msmt17.pt

--source(영상): /data/yolov5_new/pig_sample.mp4

—save-txt: 평가 지표



### 최종 결과물
<img width="80%" src="https://user-images.githubusercontent.com/74126695/210139608-67e100a0-75af-46e2-bb34-4c7518ae3523.gif"/>
