---
title: mx-maskrcnn环境搭建
date: 2019-09-06 00:00:00
categories:
  - 深度学习
tags:
  - Python
---
从<https://hub.docker.com/> 选取所需的镜像

下载caffe2 镜像  
docker pull caffe2/caffe2 :snapshot-py2-cuda8.0-cudnn7-ubuntu16.04  
Cannot open your terminal ‘/dev/pts/0’错误原因解决  
可以使用script命令来记录这个终端会话,执行  
script /dev/null  
screen -S caius  
docker 分配  
<http://www.cnblogs.com/codeaaa/p/9041533.html>  
<https://blog.csdn.net/u013948858/article/details/78429954（有效）>  
docker run -it -v /media/:/media/ —name=mxcaius —runtime=nvidia 89f57a4ade86 /bin/bash  
docker ubuntu源卡主，解决措施：  
mv source  
改完之后改回去  
mv sources.list.d.odd sources.list.d  
需要BLAS库，可以安装ATLAS、OpenBLAS、MKL，我安装的是atlas  
sudo apt-get install libatlas-base-dev  
安装opencv库  
pip install opencv-python  
sudo apt-get install libopencv-dev  
安装Python包  
cd python;  
python setup.py install  
apt-get install python-numpy  
odules/imgproc/src/resize.cpp:3596: error: (-215:Assertion failed) func != 0 in function ‘resize’

numpy 1.14  
setuptools和numpy(sudo apt-get install python-numpy)

git clone —recursive <https://github.com/apache/incubator-mxnet.git> incubator-mxnet —branch 0.11.0  
cp rcnn/CXX_OP/* incubator-mxnet/src/operator/  
cd incubator-mxnet  
make -j USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1  
cd ..  
make  
bash scripts/train_alternate.sh  
make  
caius@sugon:~$ echo -n “cvlab1205” |md5sum  
d0599e86d6134fee87bcf017ddca1990

然后我们使用docker ps查看到该容器信息，接下来就使用docker attach进入该容器  
可形变卷积

IndexError: list index out of range 

self.class_id = [0, 1] 

imdb = eval(dataset)(image_set, root_path, dataset_path)

[‘train’]imagesetIcdar2015/media/data1/caius/dataset  
[‘train’]imagesetIcdar2015/media/data1/caius/dataset

icdar2015_train gt roidb loaded from model/res50-fpn/icdar2015/alternate/cache/icdar2015_train_gt_roidb.pkl  
OpenCV Error: Assertion failed (func != 0) in resize, file /io/opencv/modules/imgproc/src/imgwarp.cpp, line 3370  
Traceback (most recent call last):  
File “train_alternate_mask_fpn.py”, line 118, in   
main()  
File “train_alternate_mask_fpn.py”, line 115, in main  
args.rcnn_epoch, args.rcnn_lr, args.rcnn_lr_step)  
File “train_alternate_mask_fpn.py”, line 61, in alternate_train  
vis=False, shuffle=False, thresh=0)  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/tools/test_rpn.py”, line 63, in test_rpn  
imdb_boxes = generate_proposals(predictor, test_data, imdb, vis=vis, thresh=thresh)  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/core/tester.py”, line 61, in generate_proposals  
for im_info, data_batch in test_data:  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/core/loader.py”, line 60, in next  
self.get_batch()  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/core/loader.py”, line 83, in get_batch  
data, label, im_info = get_rpn_testbatch(roidb)  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/io/rpn.py”, line 32, in get_rpn_testbatch  
imgs, roidb,masks = get_image(roidb)  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/io/image.py”, line 99, in get_image  
mask, _ = resize(mask, target_size, max_size)  
File “/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss/rcnn/io/image.py”, line 138, in resize  
im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)  
cv2.error: /io/opencv/modules/imgproc/src/imgwarp.cpp:3370: error: (-215) func != 0 in function resize

root@d59236d7a683:/media/data1/caius/mx-maskrcnn-original-std-broadcast-2-maskloss/mx-maskrcnn-original-std-broadcast-2-maskloss# 
