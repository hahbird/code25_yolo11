import os
from ultralytics import YOLO

if __name__ == '__main__':
    yaml_yolov11s = 'ultralytics/cfg/models/11/yolo11s.yaml'
    yaml_SE_v11 = 'ultralytics/cfg/models/11/det_self/attention/yolo11s-attention-SE.yaml'

    # 替换一下变量名即可
    model_yaml = yaml_SE_v11
    # 模型加载
    model = YOLO(model_yaml)
    # 数据集路径的yaml文件
    data_path = r'config\traindata.yaml'
    # 以yaml文件的名字进行命名
    name = os.path.basename(model_yaml).split('.')[0]
    # 文档中对参数有详细的说明
    model.train(data=data_path,             # 数据集
                imgsz=640,                  # 训练图片大小
                epochs=200,                 # 训练的轮次
                batch=2,                    # 训练batch
                workers=0,                  # 加载数据线程数
                device='cpu',                 # 使用显卡
                optimizer='SGD',            # 优化器
                project='runs/train',       # 模型保存路径
                name=name,                  # 模型保存命名
                )
