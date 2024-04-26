# 如何实现一个基于YOLOv8的人群计数系统

这个是我的毕设课题（的一部分），写一篇文章小小地记录一下  
ps：不用自己费劲巴拉选题的感觉真好……

#### 目录
- [YOLO系列简介](#YOLO系列简介)
- [环境准备](#环境准备)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型部署](#模型部署)
- [最终效果](#最终效果)

### YOLO系列简介

这部分内容有点太多了，请自助^_^  
找到一篇很NB的综述：***A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS***  
`https://arxiv.org/abs/2304.00501`

### 环境准备

安装Anaconda，打开图形用户界面的PyCharm，新建项目。  
版本参考：我的Anaconda内的PyCharm版本-2023.3。IDE版本不同的话，设置里的布局会有点点不一样\~  
点击设置-项目-Python解释器-添加解释器-添加本地解释器-系统解释器-读取本地文件找到..\anaconda3\pythonw.exe-一路确定  
恭喜你，你现在可以在PyCharm里使用Anaconda环境里已经装好的所有依赖了\~  
  
接下来安装一些要用到的库。PyCharm中打开底部的终端进行安装  
python库安装命令：pip install XXX  
(反正程序写好之后等它解释或者运行，它报什么包缺失就安装什么就可以了。大部分的库conda已经帮忙装好了所以很轻松）  
Anaconda内安装opencv-python、opencv-contrib-python  
  
安装ultralytics  
这里遇到一个坑，ultralytic库是会经常更新的，如果用的是旧的版本，运行之后会报“Import error:connot import name 'YOLO' from 'ultralytics'(unkown location)”  
这个时候需要更新库，执行pip install -U ultralytics即可解决。  

### 数据集准备

#### 准备你的数据集
勤快一点可以自己出去拍些照片\~标注工具用LabelImg，按快捷键w新建标注框，完成后选择Yolo格式导出即可\~  
注意标注文件（txt格式）需要和对应的图片名称一致  
用现成的标注好的数据集也可以，比如众所周知的COCO数据集等等。  

#### 分割数据集
接下来要分割数据集，把一坨巨大的数据集按比例随机分成train、test、valid三个部分~相关python脚本网上都有哈，很容易就能找到，也很容易跑通  
分割完，你就得到了一份新鲜的可以用于训练的数据集  

### 模型训练

#### 数据集配置
本人的小小轻薄本支撑不住大模型的狠狠♂使用，所以只好寻求服务器大大的帮助QAQ 
我用的是OpenBayes，它的界面很美观，功能很完备。  
而且它便宜。  
就是这样，所以我选择了这个平台\~  
在数据仓库-数据集模块，选择“创建新数据集”。上传过的可以在数据集详情点击“创建空版本”。  
点击“上传至当前目录”，可以直接把数据集压缩上传到网站，网站会帮你自动解压到云端数据集\~非常方便\~  

#### 算力容器配置
创建新容器，取一个炫酷的名字，再绑定刚刚配置好的数据集。算力选RTX4090，就非常够用了（还便宜，性价比之选）。审核然后执行，等待它加载完成，就ok了\~

#### YOLOv8配置
打开终端，下载YOLOv8模型。
```
git clone https://github.com/ultralytics/ultralytics
```
安装依赖库
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
这里有一个小坑，OpenBayes里面没有requirement.txt，你要自己导入一个。如果你找到的requirement.txt内容里有网址，要删掉。  
这是因为这个requirement.txt是从PyCharm导出的，那些网址是PyCharm的内部源。  
还有一个小坑，imageio要求pillow版本在[ 8.3.2，10.1.0 )。报错了需要降低pillow版本\~  
```
pip install pillow==8.3.2
```
安装ultralytics
```
 pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple/```
```
在home文件夹下创建.yaml文件，名字随便取，文件内容如下：
```yaml
#data
train: /openbayes/input/input0/train
val: /openbayes/input/input0/val
test: /openbayes/input/input0/test

#classes
names:
    0: 【你在LabelImg设置的标签名】
    1: ……
```
然后在home文件夹下创建train.py文件，内容在ultralytics的文档里有\~  
  
然后在终端运行train.py，就会开始训练了，耐心等待一下\~

### 模型部署

#### 加载模型
训练完成后，终端上的信息会告诉你，效果最好的那一代的模型文件best.pt存在哪个位置。  
把它下载下来，存在你的PyCharm项目的文件夹下，就可以在你的项目里使用它了。  
  
加载模型的语句如下：

```python
model = YOLO('best.pt')
```

#### 实现标注
supervision可以帮助我们框出每一个识别到的物体并打上标签。它还有更多功能，详情参阅官网`https://supervision.roboflow.com/`。  
终端安装supervision，即可开始使用。  
如果想要实现对物体的标注，参考代码如下：
```python
# 在这之前，你需要先读入图片
# 注释器初始化
bounding_box_annotator = sv.BoundingBoxAnnotator( 
    color=sv.Color(r=255, g=0, b=0), thickness=3) 
label_annotator = sv.LabelAnnotator(
    color=sv.Color(r=255, g=255, b=255), text_color=sv.Color(r=255, g=0, b=128), text_scale=3, text_thickness=3)
# 标签初始化
result = model(self.frame)[0]
image = self.frame
detections = sv.Detections.from_ultralytics(result)
labels = [
    model.model.names[class_id]
    for class_id
    in detections.class_id
]
# 标注，image为你想要标注的图片
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)
cv2.imshow("Pic", annotated_image)
```
  
如果想知道图片上有多少个目标物体，可以用ultralytics提供的接口：
```python
# 设置识别范围，当然你也可以通过supervision实现指定范围内识别
region_points = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
# 计数器初始化
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=1)
tracks = model.track(self.frame, persist=True)
self.frame = counter.start_counting(self.frame, tracks)
# 得到计数（指定范围内/外）
in_count = counter.in_count
out_count = counter.out_count
```
### 最终效果
最终的结果是酱婶滴\~
![YOLOv8result](https://raw.githubusercontent.com/b1ueandme/b1ueandme.github.io/blob/4ba8f43051cdaccc1d135eb257386ea7785cc476/images/2024-04-26-Crowd_Counting_With_YOLOv8/YOLOv8result.png)
