import traceback
from datetime import datetime
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor

from UI import Ui_MainWindow

from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog

import cv2
import os
import numpy as np

from ultralytics import YOLO

from tool.parser import get_config
from tool.tools import draw_info, result_info_format, format_data, writexls, writecsv, resize_with_padding


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, cfg=None):
        super().__init__()
        self.frame_number = 0
        self.comboBox_index = None
        self.results = []
        self.result_img_name = None
        self.setupUi(self)

        # 根据config配置文件更新界面配置
        self.init_UI_config()
        self.start_type = None
        self.img = None
        self.img_path = None
        self.video = None
        self.video_path = None
        # 绘制了识别信息的frame
        self.img_show = None
        self.sign = True

        self.result_info = None

        self.chinese_name = chinese_name

        # 获取当前工程文件位置
        self.ProjectPath = os.getcwd()
        self.comboBox_text = '所有目标'
        run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # 保存所有的输出文件
        self.output_dir = os.path.join(self.ProjectPath, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        result_time_path = os.path.join(self.output_dir, run_time)
        os.mkdir(result_time_path)

        # 保存txt内容
        self.result_txt = os.path.join(result_time_path, 'result.txt')
        with open(self.result_txt, 'w') as result_file:
            result_file.write(str(['序号', '图片名称', '录入时间', '识别结果', '目标数目', '用时', '保存路径'])[1:-1])
            result_file.write('\n')

        # 保存绘制好的图片结果
        self.result_img_path = os.path.join(result_time_path, 'img_result')
        os.mkdir(self.result_img_path)

        # 默认选择为所有目标
        self.comboBox_value = '所有目标'

        self.number = 1
        self.RowLength = 0
        self.consum_time = 0
        self.input_time = 0

        # 打开图片
        self.pushButton_img.clicked.connect(self.open_img)
        # 打开文件夹
        self.pushButton_dir.clicked.connect(self.open_dir)
        # 打开视频
        self.pushButton_video.clicked.connect(self.open_video)
        # 打开摄像头
        self.pushButton_camera.clicked.connect(self.open_camera)
        # 绑定开始运行
        self.pushButton_start.clicked.connect(self.start)
        # 导出数据
        self.pushButton_export.clicked.connect(self.write_files)

        self.comboBox.activated.connect(self.onComboBoxActivated)
        self.comboBox.mousePressEvent = self.handle_mouse_press

        # 表格点击事件绑定
        self.tableWidget_info.cellClicked.connect(self.cell_clicked)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.image_files = []
        self.current_index = 0

    def init_UI_config(self):
        """
        根据config.yaml中的配置，更新界面
        """
        # 更新界面标题
        self.setWindowTitle(title)
        # 更新 label_title 的标题文本
        self.label_title.setText(label_title)
        # 更新title字体颜色
        self.label_title.setStyleSheet("color: rgb(%s);" % title_color)
        # 更新label_img的尺寸
        # 更新背景图片
        self.setStyleSheet("#centralwidget {background-image: url('%s')}" % background_img)
        # 更新主图
        self.label_img.setPixmap(QtGui.QPixmap(zhutu2))

        self.pushButton_start.setStyleSheet(
            "background-color: rgb(%s); border-radius: 15px; color: rgb(%s); " % (start_button_bg, start_button_font))
        self.pushButton_export.setStyleSheet(
            "background-color: rgb(%s); border-radius: 15px; color: rgb(%s);" % (export_button_bg, export_button_font))
        # 左侧控制区域颜色
        # self.label_control.setStyleSheet("background-color: rgba(%s); border-radius: 15px;" % label_control_color)
        self.label_img.setStyleSheet("background-color: rgba(%s); border-radius: 15px;" % label_img_color)
        # 设置 tableWidget 表头 的样式
        header_style_sheet = """
                    QHeaderView::section {{
                        background-color: rgb({header_background_color});
                        color: {header_color};
                    }}
                    """.format(
            header_background_color=header_background_color,
            header_color=header_color
        )
        self.tableWidget_info.horizontalHeader().setStyleSheet(header_style_sheet)

        # 设置 tableWidget_info 的样式
        table_widget_info_style_sheet = """
                    QTableWidget {{
                        background-color: rgb({background_color});
                    }}
                    QTableView::item:hover{{
                        background-color:rgb({item_hover_background_color});
                    }}
                    """.format(
            background_color=background_color,
            item_hover_background_color=item_hover_background_color,
        )
        self.tableWidget_info.setStyleSheet(table_widget_info_style_sheet)

        # 设置表格每列宽度
        for column, width in enumerate(column_widths):
            self.tableWidget_info.setColumnWidth(column, width)

    def cell_clicked(self, row, column):
        """
        列表 单元格点击事件
        """
        self.update_comboBox_default()

        result_info = {}
        # 判断此行是否有值
        if self.tableWidget_info.item(row, 1) is None:
            return

        # 图片路径
        self.img_path = self.tableWidget_info.item(row, 1).text()
        # 识别结果
        self.results = eval(self.tableWidget_info.item(row, 3).text())
        # 保存路径
        self.result_img_name = self.tableWidget_info.item(row, 6).text()
        self.img_show = cv2.imdecode(np.fromfile(self.result_img_name, dtype=np.uint8), -1)

        if len(self.results) > 0:
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
        else:
            box = [0, 0, 0, 0]
            score = 0
            cls_name = '无目标'
        # 格式拼接
        result_info = result_info_format(result_info, box, score, cls_name)
        self.get_comboBox_value(self.results)
        self.show_all(self.img_show, result_info)

    def handle_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            # 控制选择下拉列表时，是否暂停识别
            # self.sign = False
            # 清空列表
            self.comboBox.clear()
            if type(self.comboBox_value) == str:
                self.comboBox_value = [self.comboBox_value]
            self.comboBox.addItems(self.comboBox_value)
        QComboBox.mousePressEvent(self.comboBox, event)

    def onComboBoxActivated(self):
        """
        点击下拉列表
        """
        self.sign = True
        # 选择的值
        comboBox_text = self.comboBox.currentText()
        # 值对应的索引
        self.comboBox_index = self.comboBox.currentIndex()
        result_info = {}

        if len(self.results) == 0:
            print('图片中无目标！')
            QMessageBox.information(self, "信息", "图片中无目标", QMessageBox.Yes)
            return
        # 所有目标，默认显示结果中的第一个
        if comboBox_text == '所有目标':
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
            lst_info = self.results
        else:
            # 通过索引确定选择的目标对象
            select_result = self.results[self.comboBox_index - 1]
            box = select_result[2]
            cls_name = select_result[0]
            score = select_result[1]
            lst_info = [[cls_name, score, box]]

        # 格式拼接
        result_info = result_info_format(result_info, box, score, cls_name)

        self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.img_show = draw_info(self.img, lst_info)
        self.show_all(self.img_show, result_info)

    def show_frame(self, img):
        self.update()
        if img is not None:
            # 填充颜色
            shrink = resize_with_padding(img, self.label_img.width(), self.label_img.height(),
                                         [padvalue[2], padvalue[1], padvalue[0]])
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
            QtImg = QtGui.QImage(shrink[:], shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.label_img.setPixmap(QtGui.QPixmap.fromImage(QtImg))

    def open_img(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            # 选择文件  ;;All Files (*)
            self.img_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                  "JPEG Image (*.jpg);;PNG Image (*.png);;JFIF Image (*.jfif)")
            if self.img_path == "":  # 未选择文件
                self.start_type = None
                return

            self.img_name = os.path.basename(self.img_path)
            # 显示相对应的文字
            self.label_img_path.setText(" " + self.img_path)
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 打开摄像头")

            self.start_type = 'img'
            # 读取中文路径下图片
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 显示原图
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_dir(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            self.img_path_dir = QFileDialog.getExistingDirectory(None, "选择文件夹")
            if not self.img_path_dir:
                self.start_type = None
                return

            self.start_type = 'dir'
            # 显示相对应的文字
            self.label_dir_path.setText(" " + self.img_path_dir)
            self.label_img_path.setText(" 选择图片文件")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 打开摄像头")

            self.image_files = [file for file in os.listdir(self.img_path_dir) if file.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

            if not self.image_files:
                QMessageBox.information(self, "信息", "文件夹中没有符合条件的图片", QMessageBox.Yes)
                return

            self.current_index = 0
            self.img_path = os.path.join(self.img_path_dir, self.image_files[self.current_index])
            self.img_name = self.image_files[self.current_index]

            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_video(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            # 选择文件
            self.video_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                    "mp4 Video (*.mp4);;avi Video (*.avi)")
            if not self.video_path:
                self.start_type = None
                return

            self.start_type = 'video'
            # 显示相对应的文字
            self.label_video_path.setText(" " + self.video_path)
            self.label_img_path.setText(" 选择图片文件")
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_camera_path.setText(" 打开摄像头")

            self.video_name = os.path.basename(self.video_path)
            self.video = cv2.VideoCapture(self.video_path)
            # 读取第一帧
            ret, self.img = self.video.read()
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_camera(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            if self.label_camera_path.text() == ' 打开摄像头' or self.label_camera_path.text() == ' 摄像头已关闭':
                self.start_type = 'camera'
                # 显示相对应的文字
                self.label_img_path.setText(" 选择图片文件")
                self.label_dir_path.setText(" 选择图片文件夹")
                self.label_video_path.setText(" 选择视频文件")
                self.label_camera_path.setText(" 摄像头已打开")

                self.video_name = camera_num
                self.video = cv2.VideoCapture(self.video_name)
                ret, self.img = self.video.read()
                self.show_frame(self.img)
            elif self.label_camera_path.text() == ' 摄像头已打开':
                # 修改文本为开始运行
                self.pushButton_start.setText("开始运行 >")
                self.label_camera_path.setText(" 摄像头已关闭")

        except Exception as e:
            traceback.print_exc()

    def show_all(self, img, info):
        '''
        展示所有的信息
        '''
        self.show_frame(img)
        self.show_info(info)

    def start(self):
        self.update_comboBox_default()
        try:
            if self.start_type == None:
                QMessageBox.information(self, "信息", "请先选择输入类型！", QMessageBox.Yes)
                return
            if self.start_type == 'img':
                # 读取中文路径下图片
                self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                _, result_info = self.predict_img(self.img)

                self.show_all(self.img_show, result_info)

            elif self.start_type in ['dir', 'video', 'camera']:
                if self.pushButton_start.text() == '开始运行 >':

                    self.timer.start(20)  # 每20毫秒处理一张图片或一帧
                    self.pushButton_start.setText("结束运行 >")

                    if self.start_type == 'camera':
                        self.label_camera_path.setText(" 摄像头已打开")

                elif self.pushButton_start.text() == '结束运行 >':
                    self.pushButton_start.setText("开始运行 >")
                    self.timer.stop()

                    if self.start_type == 'camera':
                        self.label_camera_path.setText(" 摄像头已关闭")
                        self.video.release()
        except Exception:
            traceback.print_exc()

    def update_frame(self):
        if self.start_type == 'dir':
            # 检查是否处理完文件夹中的所有图像
            if self.current_index >= len(self.image_files):
                self.pushButton_start.setText("开始运行 >")
                self.timer.stop()
                self.frame_number = 0  # 重置帧计数器
                if self.current_index >= len(self.image_files):
                    QMessageBox.information(self, "信息", "此文件夹已识别完", QMessageBox.Yes)
                return

            # 获取当前图像的名称和路径
            self.img_name = self.image_files[self.current_index]
            self.img_path = os.path.join(self.img_path_dir, self.img_name)
            # 读取图像并解码
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
            # 更新索引以处理下一张图像
            self.current_index += 1

        elif self.start_type in ['video', 'camera']:
            if self.frame_number == 0 and self.start_type == 'video':
                # 对于视频，处理第一帧
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, self.img = self.video.read()
                if ret:
                    # 获取当前帧号
                    frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                    # 设置图像名称
                    self.img_name = f"{self.video_name}_{frame_number}.jpg"
                    self.img_path = self.video_path
                    self.frame_number = 0  # 重置帧计数器
                else:
                    # 如果读取失败，停止计时器并释放视频资源
                    self.pushButton_start.setText("开始运行 >")
                    self.timer.stop()
                    self.video.release()
                    self.frame_number = 0  # 重置帧计数器
                    QMessageBox.information(self, "信息", "视频识别已完成", QMessageBox.Yes)
                    return
            else:
                # 读取下一帧
                ret, self.img = self.video.read()
                if not ret:
                    # 如果读取失败，停止计时器并释放视频资源
                    self.pushButton_start.setText("开始运行 >")
                    self.timer.stop()
                    self.video.release()
                    self.frame_number = 0  # 重置帧计数器
                    if self.start_type == 'video':
                        QMessageBox.information(self, "信息", "视频识别已完成", QMessageBox.Yes)
                    elif self.start_type == 'camera':
                        self.label_camera_path.setText(" 摄像头已关闭")
                        QMessageBox.information(self, "信息", "摄像头关闭", QMessageBox.Yes)
                    return
                if self.start_type == 'video':
                    # 获取当前帧号并设置图像名称
                    frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                    self.img_name = f"{self.video_name}_{frame_number}.jpg"
                    self.img_path = self.video_path
                elif self.start_type == 'camera':
                    # 对于摄像头，增加帧号并设置图像名称
                    self.frame_number += 1
                    self.img_name = f"camera_{self.frame_number}.jpg"
                    self.img_path = 'camera'

        # 进行图像预测
        results, result_info = self.predict_img(self.img)
        # 显示识别结果
        self.show_all(self.img_show, result_info)
        if self.start_type == 'video':
            # 对于视频，增加帧号以处理下一帧
            self.frame_number += 1

    def predict_img(self, img):
        # 初始化结果信息字典
        result_info = {}
        # 记录开始时间以计算处理时间
        t1 = time.time()
        # 设置结果图像的路径
        self.result_img_name = os.path.join(self.result_img_path, self.img_name)
        # 模型识别
        self.results = yolo.predict(img, imgsz=imgsz, conf=conf_thres, device=device, classes=classes)
        # 整理格式
        self.results = format_data(self.results)
        # 计算并记录消耗时间
        self.consum_time = str(round(time.time() - t1, 2)) + 's'
        # 记录输入时间
        self.input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将结果写入文件
        with open(self.result_txt, 'a+') as result_file:
            result_file.write(
                str([self.number, self.img_path, self.input_time, self.results, len(self.results), self.consum_time,
                     self.result_img_name])[1:-1])
            result_file.write('\n')

        # 显示识别信息表格
        self.show_table()
        # 增加编号
        self.number += 1
        # 获取下拉列表的值
        self.get_comboBox_value(self.results)

        if len(self.results) > 0:
            # 如果有识别结果，获取第一个结果的信息
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
        else:
            # 如果无识别结果，设置默认值
            box = [0, 0, 0, 0]
            score = 0
            cls_name = '无目标'

        # 格式化结果信息
        result_info = result_info_format(result_info, box, score, cls_name)
        # 在图像上绘制识别结果
        self.img_show = draw_info(img, self.results)
        # 保存结果图像
        cv2.imencode('.jpg', self.img_show)[1].tofile(self.result_img_name)

        return self.results, result_info

    def get_comboBox_value(self, results):
        '''
        获取当前所有的类别和ID，点击下拉列表时，使用
        '''
        # 默认第一个是 所有目标
        lst = ["所有目标"]
        for bbox in results:
            cls_name = bbox[0]
            lst.append(str(cls_name))
        self.comboBox_value = lst

    def show_info(self, result):
        try:
            if len(result) == 0:
                print("未识别到目标")
                return
            cls_name = result['cls_name']
            if len(self.chinese_name) > 3:
                cls_name = self.chinese_name[cls_name]
            if len(cls_name) > 10:
                # 当字符串太长时，显示不完整
                lst_cls_name = cls_name.split('_')
                cls_name = lst_cls_name[0][:10] + '...'

            self.label_class.setText(str(cls_name))
            self.label_score.setText(str(result['score']))
            self.label_xmin_v.setText(str(result['label_xmin_v']))
            self.label_ymin_v.setText(str(result['label_ymin_v']))
            self.label_xmax_v.setText(str(result['label_xmax_v']))
            self.label_ymax_v.setText(str(result['label_ymax_v']))
            self.update()  # 刷新界面
        except Exception as e:
            traceback.print_exc()

    def update_comboBox_default(self):
        """
        将下拉列表更新为 所有目标 默认状态
        """
        # 清空内容
        self.comboBox.clear()
        # 添加更新内容
        self.comboBox.addItems([self.comboBox_text])

    def show_table(self):
        try:
            # 显示表格
            self.RowLength = self.RowLength + 1
            self.tableWidget_info.setRowCount(self.RowLength)
            for column, content in enumerate(
                    [self.number, self.img_path, self.input_time, self.results, len(self.results), self.consum_time,
                     self.result_img_name]):
                # self.tableWidget_info.setColumnWidth(3, 0)  # 将第二列的宽度设置为0，即不显示
                row = self.RowLength - 1
                item = QtWidgets.QTableWidgetItem(str(content))
                # 居中
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                # 设置字体颜色
                item.setForeground(QColor.fromRgb(column_color[0], column_color[1], column_color[2]))
                self.tableWidget_info.setItem(row, column, item)
            # 滚动到底部
            self.tableWidget_info.scrollToBottom()
        except Exception as e:
            traceback.print_exc()

    def write_files(self):
        """
        导出 excel、csv 数据
        """
        path, filetype = QFileDialog.getSaveFileName(None, "另存为", self.ProjectPath,
                                                     "Excel 工作簿(*.xls);;CSV (逗号分隔)(*.csv)")
        with open(self.result_txt, 'r') as f:
            lst_txt = f.readlines()
            data = [list(eval(x.replace('\n', ''))) for x in lst_txt]

        if path == "":  # 未选择
            return
        if filetype == 'Excel 工作簿(*.xls)':
            writexls(data, path)
        elif filetype == 'CSV (逗号分隔)(*.csv)':
            writecsv(data, path)
        QMessageBox.information(None, "成功", "数据已保存！", QMessageBox.Yes)


# UI.ui转UI.py
# pyuic5 -x UI.ui -o UI.py
if __name__ == "__main__":
    path_cfg = 'config/configs.yaml'
    cfg = get_config()
    cfg.merge_from_file(path_cfg)
    # 加载模型相关的参数配置
    cfg_model = cfg.MODEL
    weights = cfg_model.WEIGHT
    conf_thres = float(cfg_model.CONF)
    classes = eval(cfg_model.CLASSES)
    imgsz = int(cfg_model.IMGSIZE)
    device = cfg_model.DEVICE
    # 加载UI界面相关的配置
    cfg_UI = cfg.UI
    background_img = cfg_UI.background
    padvalue = cfg_UI.padvalue
    column_widths = cfg_UI.column_widths
    column_color = cfg_UI.column_color
    title = cfg_UI.title
    title_color = cfg_UI.label_title_color
    label_title = cfg_UI.label_title
    zhutu2 = cfg_UI.zhutu2
    start_button_bg = cfg_UI.start_button_bg
    start_button_font = cfg_UI.start_button_font
    export_button_bg = cfg_UI.export_button_bg
    export_button_font = cfg_UI.export_button_font
    label_img_color = cfg_UI.label_img_color
    header_background_color = cfg_UI.table_widget_info_styles.header_background_color
    header_color = cfg_UI.table_widget_info_styles.header_color
    background_color = cfg_UI.table_widget_info_styles.background_color
    item_hover_background_color = cfg_UI.table_widget_info_styles.item_hover_background_color
    # 加载通用配置
    camera_num = int(cfg.CONFIG.camera_num)
    chinese_name = cfg.CONFIG.chinese_name

    # 模型加载
    yolo = YOLO(weights)
    # 模型预热
    yolo.predict(np.zeros((300, 300, 3), dtype='uint8'), device=device)

    # 创建QApplication实例
    app = QApplication([])
    # 创建自定义的主窗口对象
    window = MyMainWindow(cfg)
    # 显示窗口
    window.show()
    # 运行应用程序
    app.exec_()
