from config import CFG
from utility import png2tensor
from model import load_model
import os, sys
import torch
from torch import nn
from tqdm import tqdm
tqdm.pandas()

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication

from GUI_interface import Ui_MainWindow
import matplotlib.pyplot as plt
import time as tm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model ("resnet50", "./models-for-unet/model_unet_resnet50_2.bin")


# GUI功能实现
class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cwd = os.getcwd()
        self.setWindowTitle("肠胃分割软件")

        self.labelinput.setAlignment(Qt.AlignCenter)
        self.labelinput.setStyleSheet("QLabel{background:gray;}")

        self.labelresult.setAlignment(Qt.AlignCenter)
        self.labelresult.setStyleSheet("QLabel{background:gray;}")

        self.textBrowser.setStyleSheet("background-color: cyan;")

        self.setWindowIcon(QIcon('./Figures/background.png'))

        # 设置背景图片
        palette1 = QPalette()
        pix = QPixmap('./Figures/background.png')
        pix = pix.scaled(self.width(), self.height())
        palette1.setBrush(self.backgroundRole(), QBrush(pix))
        self.setPalette(palette1)

        # self.btnInput.setIcon(QIcon(r'icon/open_img.png'))
        self.btnInput.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")

        self.btnmultiInput.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")

        self.btnonlyLargeBowel.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")

        self.btnonlySmallBowel.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")

        self.btnonlyStomach.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")

        # self.btnTest.setIcon(QIcon(r'icon/start.png'))
        self.btnTest.setStyleSheet("QPushButton{color:black}"
                                    "QPushButton:hover{color:red}"
                                    "QPushButton{background-color:lightgreen}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:10px}"
                                    "QPushButton{padding:2px 4px}")

        self.choice = 0
        self.tested = False

        self.img = None
        self.scaled_img = None
        self.left_click = False
        self.wheel_flag = False

        self.scale = 1
        self.old_scale = 1
        self.point = QPoint(610, 70)
        self.x = 610
        self.y = 70
        self.new_height = 224
        self.new_width = 224
        # 鼠标移动坐标指针
        self.mouse_mv_x = ""
        self.mouse_mv_y = ""
        self.x1 = 0
        self.y1 = 0

        self.imgx = 610
        self.imgy = 70


    def btnTest_Pressed(self):
        if not hasattr(self, "captured"):
            # print("没有输入图像")
            # self.textBrowser.setPlainText("没有输入图像")
            return
        self.textBrowser.append("图像检测中...")


    def btnInput_Clicked(self):
        '''
        从本地读取图片
        '''
        global fname
        # 打开文件选取对话框
        filename, _ = QFileDialog.getOpenFileName(self, '打开图片', "", "*.png;;*.jpg;;All Files(*)")
        if filename:
            self.choice = 1
            img = png2tensor(filename)
            img = img.permute((1, 2, 0)).numpy() * 255.0
            img = img.astype('uint8')
            rows, cols, channels = img.shape
            bytesPerLine = channels * cols

            QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelinput.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelinput.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            return
        fname = filename
        self.textBrowser.setPlainText("成功打开图片")
        self.tested = False


    def btnmultiInput_Clicked(self):
        '''
        从本地读取图片
        '''
        global fname
        # 打开文件选取对话框
        filename, _ = QFileDialog.getOpenFileNames(self, '打开多个图片', self.cwd, "*.png;;*.jpg;;All Files(*)")
        if filename:
            self.choice = 2
        else:
            return
        fname = filename
        self.textBrowser.setPlainText("成功打开图片")


    def btnTest_Clicked(self):
        '''
        test
        '''
        global fname
        # 如果没有捕获图片，则不执行操作
        if self.choice == 0:
            print("没有输入图像")
            self.textBrowser.setPlainText("没有输入图像")
            return
        print("start")
        # -*- coding: utf-8 -*-

        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['font.serif'] = ['KaiTi']

        a = tm.time()

        if self.choice == 1:           # 单张图片分割
            self.labelresult.setPixmap(QPixmap(""))  # 移除label上的图片
            image = png2tensor(fname)
            img = image.permute((1, 2, 0)).numpy() * 255.0
            img = img.astype('uint8')
            image = image.reshape(-1, 3, 224, 224)
            # image = torch.tensor(image).to(torch.float32)

            image = image.to(CFG.device, dtype=torch.float)
            with torch.no_grad():
                preds = model(image)
                preds = (nn.Sigmoid()(preds) > 0.5).double().cpu().detach()
            preds = preds[0].permute((1, 2, 0)).numpy() * 255.0
            preds = preds.astype('uint8')

            plt.imshow(img, cmap='bone')
            plt.imshow(preds, alpha=0.5)
            handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                       [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(handles, labels)  # 标签与掩码颜色相对应
            plt.axis('off')

            # return output
            plt.savefig("./results/res.png", bbox_inches='tight', pad_inches=-0.1)

            getimg = cv2.imread("./results/res.png")
            self.img = "./results/res.png"

            for i in range(3):
                pred = preds.copy()
                if i == 0:
                    pred[:, :, 1:] = 0
                    plt.imshow(img, cmap='bone')
                    plt.imshow(pred, alpha=0.5)
                    handle = [Rectangle((0, 0), 1, 1, color=(0.667, 0.0, 0.0))]
                    label = ["Large Bowel"]
                elif i == 1:
                    pred[:, :, 0] = 0
                    pred[:, :, 2] = 0
                    plt.imshow(img, cmap='bone')
                    plt.imshow(pred, alpha=0.5)
                    handle = [Rectangle((0, 0), 1, 1, color=(0.0, 0.667, 0.0))]
                    label = ["Small Bowel"]
                else:
                    pred[:, :, 0:2] = 0
                    plt.imshow(img, cmap='bone')
                    plt.imshow(pred, alpha=0.5)
                    handle = [Rectangle((0, 0), 1, 1, color=(0.0, 0.0, 0.667))]
                    label = ["Stomach"]
                plt.legend(handle, label)  # 标签与掩码颜色相对应
                plt.axis('off')
                # return output
                plt.savefig(f"./results/res{i}.png", bbox_inches='tight', pad_inches=-0.1)

            rows, cols, channels = getimg.shape
            bytesPerLine = channels * cols
            QImg = QImage(getimg.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.tested = True
            self.initialize()

        elif self.choice == 2:                  # 多张图片分割
            self.textBrowser.append(f"共{len(fname)}张图片")
            for i in range(len(fname)):
                image = png2tensor(fname[i])
                img = image.permute((1, 2, 0)).numpy() * 255.0
                img = img.astype('uint8')
                image = image.reshape(-1, 3, 224, 224)
                # image = torch.tensor(image).to(torch.float32)

                image = image.to(CFG.device, dtype=torch.float)
                with torch.no_grad():
                    preds = model(image)
                    preds = (nn.Sigmoid()(preds) > 0.5).double().cpu().detach()
                preds = preds[0].permute((1, 2, 0)).numpy() * 255.0
                preds = preds.astype('uint8')

                # Show area outside image boundaries.
                _, ax = plt.subplots(1, 2, figsize=(16, 16))
                # ax[0].set_ylim(height + 10, -10)
                # ax[0].set_xlim(-10, width + 10)
                ax[0].axis('off')
                ax[0].imshow(img, cmap='bone')

                ax[1].imshow(img, cmap='bone')
                ax[1].imshow(preds, alpha=0.5)
                handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                           [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
                labels = ["Large Bowel", "Small Bowel", "Stomach"]
                ax[1].legend(handles, labels)  # 标签与掩码颜色相对应
                ax[1].axis('off')

                # return output
                plt.savefig(f"./results/image{i}.png", bbox_inches='tight', pad_inches=-0.1)

        b = tm.time()
        time = b - a
        self.textBrowser.append("检测用时：" + str(time) + "s")
        print("ending")

        self.textBrowser.append("检测完成")


    def btnonlyLargeBowel_Clicked(self):
        # 如果没有检测图片，则不执行操作
        if self.tested == False:
            print("没有检测图像")
            self.textBrowser.setPlainText("没有检测图像")
            self.labelresult.setPixmap(QPixmap(""))  # 移除label上的图片
            return
        print("start")
        # -*- coding: utf-8 -*-

        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['font.serif'] = ['KaiTi']

        getimg = cv2.imread("./results/res0.png")

        rows, cols, channels = getimg.shape
        bytesPerLine = channels * cols
        QImg = QImage(getimg.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def btnonlySmallBowel_Clicked(self):
        # 如果没有检测图片，则不执行操作
        if self.tested == False:
            print("没有检测图像")
            self.textBrowser.setPlainText("没有检测图像")
            self.labelresult.setPixmap(QPixmap(""))  # 移除label上的图片
            return
        print("start")
        # -*- coding: utf-8 -*-

        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['font.serif'] = ['KaiTi']

        getimg = cv2.imread("./results/res1.png")

        rows, cols, channels = getimg.shape
        bytesPerLine = channels * cols
        QImg = QImage(getimg.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def btnonlyStomach_Clicked(self):
        # 如果没有检测图片，则不执行操作
        if self.tested == False:
            print("没有检测图像")
            self.textBrowser.setPlainText("没有检测图像")
            self.labelresult.setPixmap(QPixmap(""))  # 移除label上的图片
            return
        print("start")
        # -*- coding: utf-8 -*-

        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['font.serif'] = ['KaiTi']

        getimg = cv2.imread("./results/res2.png")

        rows, cols, channels = getimg.shape
        bytesPerLine = channels * cols
        QImg = QImage(getimg.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def Repaint(self):
        getimg = cv2.imread(self.img)
        rows, cols, channels = getimg.shape
        bytesPerLine = channels * cols
        QImg = QImage(getimg.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        self.old_scale = self.scale
        self.x, self.y = event.x(), event.y()
        if self.x >= self.imgx and self.y >= self.imgy and (self.x <= self.imgx + 384 * self.scale) and (
                self.y <= self.imgy + 384 * self.scale):
            if self.tested:
                self.wheel_flag = True
                # 获取当前鼠标相对于view的位置
                if angleY > 0:
                    self.scale *= 1.08
                else:  # 滚轮下滚
                    self.scale *= 0.92
                if self.scale < 0.3:
                    self.scale = 0.3
                # self.adjustSize()
                # self.update()
                self.imgx = self.imgx + (self.old_scale - self.scale) * (self.x - self.imgx) / self.old_scale
                self.imgy = self.imgy + (self.old_scale - self.scale) * (self.y - self.imgy) / self.old_scale
                self.labelresult.setGeometry(QtCore.QRect(self.imgx-610, self.imgy-70, 384 * self.scale, 384 * self.scale))
                self.Repaint()

    def mouseMoveEvent(self, e):
        if self.left_click:  # 鼠标被点击信号，为True表示鼠标点击事件触发
            self.x1 = e.x()
            self.y1 = e.y()
            if self.x1 >= self.imgx and self.y1 >= self.imgy and (self.x1 <= self.imgx + 384 * self.scale) and (
                    self.y1 <= self.imgy + 384 * self.scale):
                if self.tested:
                    if self.mouse_mv_x != "":  # self.mouse_mv_x的初始值为""
                        if self.mouse_mv_y != "":
                            self.imgx = self.imgx + (self.x1 - self.mouse_mv_x)
                            self.imgy = self.imgy + (self.y1 - self.mouse_mv_y)
                    self.mouse_mv_x = self.x1  # 记录拖拽时上一个点的坐标
                    self.mouse_mv_y = self.y1
                    self.labelresult.setGeometry(
                        QtCore.QRect(self.imgx-610, self.imgy-70, 384 * self.scale, 384 * self.scale))
                    self.Repaint()


    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False
            # 销毁结点指针
            self.mouse_mv_y = ""
            self.mouse_mv_x = ""

    def initialize(self):
        self.left_click = False
        self.wheel_flag = False

        self.scale = 1
        self.old_scale = 1
        self.point = QPoint(610, 70)
        self.x = 610
        self.y = 70
        self.new_height = 224
        self.new_width = 224
        # 鼠标移动坐标指针
        self.mouse_mv_x = ""
        self.mouse_mv_y = ""
        self.x1 = 0
        self.y1 = 0

        self.imgx = 610
        self.imgy = 70
        self.labelresult.setGeometry(QtCore.QRect(0, 0, 384, 384))
        self.Repaint()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())

