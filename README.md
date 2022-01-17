![](https://img.shields.io/badge/Python-3.8%2B-red)
![](https://img.shields.io/badge/Pytorch-1.6%2B-brightgreen)
![](https://img.shields.io/badge/Infrared_Small_Dim_Target_Detection-yellow)

# 基于图像分割的红外弱小目标检测算法
# Infrared Target Detection by Segmentation (Deeplearing Method)
## support **Unet** and **FCN** 
![](./sirst/test_image_0.png)
![](./sirst/train_image_0.png)

## [Usage]

### Training
```python
python main.py train
```
### Test
```python
python main.py test image_dir
```
### Evaluation
```python
python main.py evaluate
```
### Visual DataSet 
```python
python main.py vis_dl
```

## [Datasets]
- SIRST dataset is available at [SIRST](https://github.com/YimianDai/sirst).

## More Configs in main.py

## Model Weights