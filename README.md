# Intelligent-identification-of-fabric-defects
This project is a tianchi competition project. the website about this competition is:[2019广东工业智造创新大赛【赛场一】][2]
The project is based on [mmdetection][0] and the main structure of this project is : cascade_mask_rcnn_r50_fpn_1x. All of these code in based on Pytorch.
The main purpose of this project is to realize the flaw identification of the fabric.

## Dependencies

- Python 3.6
  - torch >= 1.1.0
  - torchvision 0.3
  - numpy 1.16.3
  - tqdm 4.31.1
  - pandas 0.24.0
  - Pillow 5.3.0
  - argparse 1.4.0
- CUDA 10.0
- CUDNN 7.6.1
  

  
## Prepare dataset 

- Prepare the data by running
```
python Fabric2COCO.py
python argumentation.py
```
Run `Fabric2COCO.py` will creates an dataset like coco by transform the origin fabric dataset containing 5913 train image in `../data/coco/images/train/` and a json file in  `../data/coco/annotations/`.
Run `argumentation.py` will flip and rotate the new dataset and create a arugumentation dataset. This method is base on [a simple argumentation][1]

After these operation, the structure of data will be (after unzip all zip file):
```
|–data
|-- coco
　|-- annotations
  　|-- instances_train.json
  　|-- train_rotate180.json
　|-- images
　　|-- train
　　|-- defect_Images_rotate180
|-- First_round_data
　|-- guangdong1_round1_train1_20190818
　|-- guangdong1_round1_train2_20190828
　|-- guangdong1_round1_testA_20190818

|-- guangdong1_round1_testB_20190919
```

## How to Train

The structure of network and most of the parameters is under `config.py`.
To train the model you can run
```
python train.py
```
- To predict the test data(the defalut dataset is testB, if you want change the dataset, you can change the test_path in `test.py`) you can run
```
python test.py
``` 


[0]: https://github.com/open-mmlab/mmdetection
[1]: https://tianchi.aliyun.com/notebook-ai/detail?postId=74575
[2]: https://tianchi.aliyun.com/competition/entrance/231748/introduction
