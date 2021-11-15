import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QEvent
# from plots_cv_platenumber import *
from TrackingAPI8_0915 import *   
from tqdm import tqdm
import os

import cv2
import numpy as np
import time
import datetime

from shapely.geometry import Polygon
import shutil
import glob
import re


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # change_pixmap_signal = pyqtSignal(np.ndarray)


    def __init__(self,cam_num):
        super().__init__()
        self._run_flag = True
        self.capture_flag = False

        self.fiducial_on = False

        self.state = 'ready'
        self.path = './data/'

        self.fiducial_point = [0,0,0,0]
        self.roi_point = [0,0,0,0]
       
        self.trigger_mode = 2

        self.cam_num = cam_num
    
    def camon(self,cam_width):
        # self.cap = cv2.VideoCapture(self.cam_num,cv2.CAP_V4L2)
        self.cap = cv2.VideoCapture(self.cam_num,cv2.CAP_DSHOW)
        if self.cap.isOpened():
            print(self.cap,"Webcam online.")
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            print(self.cap.get(cv2.CAP_PROP_FPS), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) , self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print(cap.set(cv2.CAP_PROP_FORMAT),cv2.CV_8UC3)
        

    def PolygonArea(self, corners):
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[i][0] * corners[j][1]
        area = abs(area) / 2.0
        return area


    def run(self):
        PTime = 0
        while self._run_flag:
            ret,self.cv_img = self.cap.read()
            if self.fiducial_on == True:
                fiducial_center, fiducial_box, angle, check = fiducial_marker(self.cv_img, self.trainKP, self.trainDesc, self.trainImg, self.im_aspect_ratio, self.im_height, self.im_width, self.im_area)
                if check == False:
                    pass
                else:
                    polya = Polygon([(self.roi_point[0], self.roi_point[1]), (self.roi_point[0]+self.roi_point[2], self.roi_point[1]), (self.roi_point[0]+self.roi_point[2], self.roi_point[1]+self.roi_point[3]), (self.roi_point[0], self.roi_point[1]+self.roi_point[3])]) 
                    polyb = Polygon([(fiducial_box[0][0][0], fiducial_box[0][0][1]), (fiducial_box[0][1][0], fiducial_box[0][1][1]), (fiducial_box[0][2][0], fiducial_box[0][2][1]), (fiducial_box[0][3][0], fiducial_box[0][3][1])]) 

                    contain_condition=polya.contains(polyb)

                    if self.state == 'ready':
                        if contain_condition == True:
                            
                            tm = datetime.datetime.today()
                            tmstring = str(tm.year) + str(tm.month) + str(tm.day) +'_'+ str(tm.hour) +'_'+ str(tm.minute) +'_'+ str(tm.second) +'_'+ str(tm.microsecond)[1]
                
                            # cv2.imwrite(self.path + 'cam' + str(i) +'_'+ tmstring + '.jpg',ll)

                            cv2.polylines(self.cv_img, fiducial_box, True, (0,0,255), 5)
                            print('shot')
                            self.state = 'shot'
                        else:
                            cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 2)
                            cv2.circle(self.cv_img,(20,20),5,(0,255,0),5)

                    elif self.state == 'shot':
                        if contain_condition == True:
                            cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 5)
                            cv2.circle(self.cv_img,(20,20),5,(0,0,255),5)
                        else:
                            cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 2)
                            self.state = 'ready'

       

            cv2.rectangle(self.cv_img,(self.roi_point[0],self.roi_point[1]),(self.roi_point[0]+self.roi_point[2],self.roi_point[1]+self.roi_point[3]),(255,255,0),2)

            cTime = time.time()
            sec = cTime - PTime
            PTime = time.time()
            fps = 1 / (sec)
            # print(fps)
            self.change_pixmap_signal.emit(self.cv_img)

        # shut down capture system 
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


form_class = uic.loadUiType("acq.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.current_cap = 0
        self.disply_width = 640
        self.display_height = 480

        self.canvas_mode0 = 'default'
        self.canvas_mode1 = 'default'
        self.state_draw_rect = False

        self.past_x = 0
        self.past_y = 0
        
        self.present_x = 0
        self.present_y = 0

        self.tabWidget_mode_0.tabBarClicked.connect(self.tabWidget_mode_0clicked)

        self.pushButton_fiducial_0.clicked.connect(self.pushButton_fiducial_0Function)
        self.pushButton_roi_0.clicked.connect(self.pushButton_roi_0Function)
        self.pushButton_fiducial_1.clicked.connect(self.pushButton_fiducial_1Function)
        self.pushButton_roi_1.clicked.connect(self.pushButton_roi_1Function)

        self.pushButton_camon.clicked.connect(self.pushButton_camonFunction)
        self.pushButton_run_fi_0.clicked.connect(self.pushButton_run_fi_0Function)
        self.pushButton_run_fi_1.clicked.connect(self.pushButton_run_fi_1Function)
        self.pushButton_run_mn.clicked.connect(self.pushButton_run_mnFunction)
        self.pushButton_path.clicked.connect(self.pushButton_pathFunction)
        self.lineEdit_path.textChanged.connect(self.lineEdit_pathFunction)

        self.horizontalSlider_focus_0.valueChanged.connect(self.horizontalSlider_focus_0Function)
        self.spinBox_focus_0.valueChanged.connect(self.spinBox_focus_0Function)
        self.horizontalSlider_exposure_0.valueChanged.connect(self.horizontalSlider_exposure_0Function)
        self.spinBox_exposure_0.valueChanged.connect(self.spinBox_exposure_0Function)
        self.horizontalSlider_brightness_0.valueChanged.connect(self.horizontalSlider_brightness_0Function)
        self.spinBox_brightness_0.valueChanged.connect(self.spinBox_brightness_0Function)
        self.horizontalSlider_contrast_0.valueChanged.connect(self.horizontalSlider_contrast_0Function)
        self.spinBox_contrast_0.valueChanged.connect(self.spinBox_contrast_0Function)
        self.horizontalSlider_saturation_0.valueChanged.connect(self.horizontalSlider_saturation_0Function)
        self.spinBox_saturation_0.valueChanged.connect(self.spinBox_saturation_0Function)
        self.horizontalSlider_sharpness_0.valueChanged.connect(self.horizontalSlider_sharpness_0Function)
        self.spinBox_sharpness_0.valueChanged.connect(self.spinBox_sharpness_0Function)
        self.pushButton_ispinit_0.clicked.connect(self.pushButton_autoip_0Function)

        self.horizontalSlider_focus_1.valueChanged.connect(self.horizontalSlider_focus_1Function)
        self.spinBox_focus_1.valueChanged.connect(self.spinBox_focus_1Function)
        self.horizontalSlider_exposure_1.valueChanged.connect(self.horizontalSlider_exposure_1Function)
        self.spinBox_exposure_1.valueChanged.connect(self.spinBox_exposure_1Function)
        self.horizontalSlider_brightness_1.valueChanged.connect(self.horizontalSlider_brightness_1Function)
        self.spinBox_brightness_1.valueChanged.connect(self.spinBox_brightness_1Function)
        self.horizontalSlider_contrast_1.valueChanged.connect(self.horizontalSlider_contrast_1Function)
        self.spinBox_contrast_1.valueChanged.connect(self.spinBox_contrast_1Function)
        self.horizontalSlider_saturation_1.valueChanged.connect(self.horizontalSlider_saturation_1Function)
        self.spinBox_saturation_1.valueChanged.connect(self.spinBox_saturation_1Function)
        self.horizontalSlider_sharpness_1.valueChanged.connect(self.horizontalSlider_sharpness_1Function)
        self.spinBox_sharpness_1.valueChanged.connect(self.spinBox_sharpness_1Function)
        self.pushButton_ispinit_1.clicked.connect(self.pushButton_autoip_1Function)


        self.radioButton_1280.clicked.connect(self.groupboxRadFunction)
        self.radioButton_960.clicked.connect(self.groupboxRadFunction)
  
        self.thread = []

        ret=glob.glob('/dev/video*')

        for video in ret:
            numbers = re.findall("\d+", video)
            self.thread.append(VideoThread(int(numbers[0])))
            print(numbers)

        if len(self.thread) == 1:
            self.thread[0].change_pixmap_signal.connect(self.update_image0)
        elif len(self.thread) == 2:
            self.thread[0].change_pixmap_signal.connect(self.update_image0)
            self.thread[1].change_pixmap_signal.connect(self.update_image1)
       
        # start the thread
        
        self.label_screen_0.mousePressEvent = self.screen_0_mousePressEvent
        self.label_screen_0.mouseMoveEvent = self.screen_0_mouseMoveEvent
        self.label_screen_0.mouseReleaseEvent = self.screen_0_mouseReleaseEvent

        self.label_screen_1.mousePressEvent = self.screen_1_mousePressEvent
        self.label_screen_1.mouseMoveEvent = self.screen_1_mouseMoveEvent
        self.label_screen_1.mouseReleaseEvent = self.screen_1_mouseReleaseEvent

        self.width = 1280

        diskLabel = './'
        total, used, free = shutil.disk_usage(diskLabel)

        print(total)
        print(used)
        print(free)


    def groupboxRadFunction(self) :
        if self.radioButton_1280.isChecked() : self.width = 1280
        elif self.radioButton_960.isChecked() : self.width = 960

    def pushButton_camonFunction(self):
        for thread in self.thread:
            thread.camon(self.width)
            thread.start()

    def tabWidget_mode_0clicked(self, index):
        for thread in self.thread:
            thread.trigger_mode = index
            print(thread.trigger_mode)

    def lineEdit_pathFunction(self):
        self.path=self.lineEdit_path.text() 

    def pushButton_pathFunction(self):
        self.path = QFileDialog.getExistingDirectory(self, "select path") + '/'
        self.lineEdit_path.setText(self.path)

    def pushButton_run_mnFunction(self):
        if self.thread.trigger_mode == 0:
            tm = datetime.datetime.today()
            tmstring = str(tm.year) + str(tm.month) + str(tm.day) +'_'+ str(tm.hour) +'_'+ str(tm.minute) +'_'+ str(tm.second) +'_'+ str(tm.microsecond)[1]


    def pushButton_run_fi_0Function(self):
        # img=cv2.imread('./samples/test.jpg')

        self.current_cap = 0

        if self.thread[self.current_cap].fiducial_on == False:
            print('Run Inference')
            self.canvas_mode0 = 'default'
            self.thread[self.current_cap].fiducial_on = True
            self.pushButton_run_fi_0.setText("&STOP")
        else:
            self.thread[self.current_cap].fiducial_on = False
            print('Stop Inference')
            self.pushButton_run_fi_0.setText("&RUN")
    
    def pushButton_run_fi_1Function(self):
        # img=cv2.imread('./samples/test.jpg')

        self.current_cap = 1


        if self.thread[self.current_cap].fiducial_on == False:
            print('Run Inference')
            self.canvas_mode1 = 'default'
            self.thread[self.current_cap].fiducial_on = True
            self.pushButton_run_fi_1.setText("&STOP")
        else:
            self.thread[self.current_cap].fiducial_on = False
            print('Stop Inference')
            self.pushButton_run_fi_1.setText("&RUN")



    def screen_0_mousePressEvent(self , event):
        self.past_x = event.pos().x()
        self.past_y = event.pos().y()
        self.state_draw_rect = True
       
    def screen_0_mouseMoveEvent(self , event):
        self.present_x = event.pos().x()
        self.present_y = event.pos().y()
        

    def screen_0_mouseReleaseEvent(self , event):

        self.current_cap = 0

        if self.thread[self.current_cap].trigger_mode == 2:
            if self.canvas_mode0 == 'fiducial' and self.state_draw_rect == True:
                self.canvas_mode0 = 'default'
                self.state_draw_rect = False
                self.thread[self.current_cap].fiducial_point= (self.past_x*2,self.past_y*2,self.present_x*2 -self.past_x*2 ,self.present_y*2 - self.past_y*2)
                self.fiducial_signal_0(self.thread[self.current_cap])
            
            if self.canvas_mode0 == 'roi' and self.state_draw_rect == True:
                self.canvas_mode0 = 'default'
                self.state_draw_rect = False
                self.thread[self.current_cap].roi_point= (self.past_x*2,self.past_y*2,self.present_x*2 -self.past_x*2 ,self.present_y*2 - self.past_y*2)
               

        self.present_x = event.pos().x()
        self.present_y = event.pos().y()
        # print(self.past_x,self.past_y,self.present_x,self.present_y)
        self.past_x = 0
        self.past_y = 0
        self.present_x = 0
        self.present_y = 0


    def fiducial_signal_0(self, thread):
        # self.lineEdit.setText(str(a))
        print(thread.fiducial_point)
        
        # self.thread.trainKP, self.thread.trainDesc, self.thread.imgMed, self.thread.im_aspect_ratio, self.thread.im_height, self.thread.im_width, self.thread.im_area,imCrop = search_feature(self.thread.cv_img,a,b)
        thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width, imCrop, thread.im_area = search_feature(thread.cv_img,thread.fiducial_point)

        fiducial_center, fiducial_box, angle, check = fiducial_marker(thread.cv_img, thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width,thread.im_area)
        print(imCrop.shape)
        imcrop_qt_img = self.convert_cv_qt(imCrop,120,120) 
        self.label_screen_fiducial_0.setPixmap(imcrop_qt_img)

        if check:
            thread.fiducial_center_offset = fiducial_center
            thread.fiducial_angle_offset = angle
            print('good')
        else:
            print('again')


    def screen_1_mousePressEvent(self , event):
        self.past_x = event.pos().x()
        self.past_y = event.pos().y()
        self.state_draw_rect = True
       
    def screen_1_mouseMoveEvent(self , event):
        self.present_x = event.pos().x()
        self.present_y = event.pos().y()
        

    def screen_1_mouseReleaseEvent(self , event):

        self.current_cap = 1

        if self.thread[self.current_cap].trigger_mode == 2:
            if self.canvas_mode1 == 'fiducial' and self.state_draw_rect == True:
                self.canvas_mode1 = 'default'
                self.state_draw_rect = False
                self.thread[self.current_cap].fiducial_point= (self.past_x*2,self.past_y*2,self.present_x*2 -self.past_x*2 ,self.present_y*2 - self.past_y*2)
                self.fiducial_signal_1(self.thread[self.current_cap])
            
            if self.canvas_mode1 == 'roi' and self.state_draw_rect == True:
                self.canvas_mode1 = 'default'
                self.state_draw_rect = False
                self.thread[self.current_cap].roi_point= (self.past_x*2,self.past_y*2,self.present_x*2 -self.past_x*2 ,self.present_y*2 - self.past_y*2)
               

        self.present_x = event.pos().x()
        self.present_y = event.pos().y()
        # print(self.past_x,self.past_y,self.present_x,self.present_y)
        self.past_x = 0
        self.past_y = 0
        self.present_x = 0
        self.present_y = 0


    def fiducial_signal_1(self, thread):
        # self.lineEdit.setText(str(a))
        print(thread.fiducial_point)
        
        # self.thread.trainKP, self.thread.trainDesc, self.thread.imgMed, self.thread.im_aspect_ratio, self.thread.im_height, self.thread.im_width, self.thread.im_area,imCrop = search_feature(self.thread.cv_img,a,b)
        thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width, imCrop, thread.im_area = search_feature(thread.cv_img,thread.fiducial_point)

        try:
            fiducial_center, fiducial_box, angle, check = fiducial_marker(thread.cv_img, thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width,thread.im_area)
            print(imCrop.shape)
            imcrop_qt_img = self.convert_cv_qt(imCrop,120,120) 
            self.label_screen_fiducial_1.setPixmap(imcrop_qt_img)

            if check:
                thread.fiducial_center_offset = fiducial_center
                thread.fiducial_angle_offset = angle
                print('good')
            else:
                print('again')
        except:
            print('again')


    def draw_rect(self,img,x1,y1,x2,y2, color):
        
        painter = QPainter(img)
        painter.setPen(QPen(color, 1, Qt.SolidLine))
        painter.drawRect(x1,y1,x2-x1,y2-y1)
    
   

    def closeEvent(self, event):
        for thread in self.thread:
            thread.stop()
            event.accept()


    def pushButton_fiducial_0Function(self):
        
            
        if self.canvas_mode0 != 'fiducial':
            self.canvas_mode0 = 'fiducial'
        else:
            self.canvas_mode0 = 'default'
    
    def pushButton_roi_0Function(self):
        
            
        if self.canvas_mode0 != 'roi':
            self.canvas_mode0 = 'roi'
        else:
            self.canvas_mode0 = 'default'



    def pushButton_fiducial_1Function(self):
        
            
        if self.canvas_mode1 != 'fiducial':
            self.canvas_mode1 = 'fiducial'
        else:
            self.canvas_mode1 = 'default'
    
    def pushButton_roi_1Function(self):
        
            
        if self.canvas_mode1 != 'roi':
            self.canvas_mode1 = 'roi'
        else:
            self.canvas_mode1 = 'default'



    def pushButton_autoip_0Function(self):
        self.current_cap = 0
        self.focus_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_FOCUS))
        self.brightness_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_BRIGHTNESS))
        self.contrast_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_CONTRAST))
        self.saturation_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SATURATION))
        self.exposure_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_EXPOSURE))
        self.sharpness_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SHARPNESS))

        print('focus : ' + str(self.focus_0))
        print('brightness : ' + str(self.brightness_0))
        print('contrast : ' + str(self.contrast_0))
        print('saturation : ' + str(self.saturation_0))
        print('exposure : ' + str(self.exposure_0))
        print('sharpness : ' + str(self.sharpness_0))

        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # AUTOFOCUS 켜기

        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_BRIGHTNESS, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_CONTRAST, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, 128/1)


    def horizontalSlider_focus_0Function(self):
        self.current_cap = 0
        self.spinBox_focus_0.setValue(self.horizontalSlider_focus_0.value())
        self.focus_0 = self.horizontalSlider_focus_0.value() *5
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_FOCUS, self.focus_0)

    def spinBox_focus_0Function(self):
        self.horizontalSlider_focus_0.setValue(self.spinBox_focus_0.value())

    #-- CAP_PROP_BRIGHTNESS, 0~255, step=1, default=128
    def horizontalSlider_brightness_0Function(self):
        self.current_cap = 0
        self.spinBox_brightness_0.setValue(self.horizontalSlider_brightness_0.value())
        self.brightness_0 = self.horizontalSlider_brightness_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_0/1)

    def spinBox_brightness_0Function(self):
        self.horizontalSlider_brightness_0.setValue(
            self.spinBox_brightness_0.value())

    #-- CAP_PROP_CONTRAST, 0~255, step=1
    def horizontalSlider_contrast_0Function(self):
        self.current_cap = 0
        self.spinBox_contrast_0.setValue(self.horizontalSlider_contrast_0.value())
        self.contrast_0 = self.horizontalSlider_contrast_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_CONTRAST, self.contrast_0/1)

    def spinBox_contrast_0Function(self):
        self.horizontalSlider_contrast_0.setValue(self.spinBox_contrast_0.value())

    #-- CAP_PROP_SATURATION, 0~255, step=1
    def horizontalSlider_saturation_0Function(self):
        self.current_cap = 0
        self.spinBox_saturation_0.setValue(
            self.horizontalSlider_saturation_0.value())
        self.saturation_0 = self.horizontalSlider_saturation_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, self.saturation_0/1)

    def spinBox_saturation_0Function(self):
        self.horizontalSlider_saturation_0.setValue(
            self.spinBox_saturation_0.value())

    #-- CAP_PROP_EXPOSURE, 3~2047, step=1, default=250
    def horizontalSlider_exposure_0Function(self):
        self.current_cap = 0
        self.spinBox_exposure_0.setValue(self.horizontalSlider_exposure_0.value())
        self.exposure_0 = self.horizontalSlider_exposure_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_0)

    def spinBox_exposure_0Function(self):
        self.horizontalSlider_exposure_0.setValue(self.spinBox_exposure_0.value())

    #-- CAP_PROP_SHARPNESS, 0~255, step=1, default=128
    def horizontalSlider_sharpness_0Function(self):
        self.current_cap = 0
        self.spinBox_sharpness_0.setValue(
            self.horizontalSlider_sharpness_0.value())
        self.sharpness_0 = self.horizontalSlider_sharpness_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness_0/1)

    def spinBox_sharpness_0Function(self): 
        self.horizontalSlider_sharpness_0.setValue(
            self.spinBox_sharpness_0.value())




    def pushButton_autoip_1Function(self):
        self.current_cap = 1
        self.focus_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_FOCUS))
        self.brightness_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_BRIGHTNESS))
        self.contrast_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_CONTRAST))
        self.saturation_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SATURATION))
        self.exposure_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_EXPOSURE))
        self.sharpness_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SHARPNESS))

        print('focus : ' + str(self.focus_1))
        print('brightness : ' + str(self.brightness_1))
        print('contrast : ' + str(self.contrast_1))
        print('saturation : ' + str(self.saturation_1))
        print('exposure : ' + str(self.exposure_1))
        print('sharpness : ' + str(self.sharpness_1))

        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # AUTOFOCUS 켜기
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_BRIGHTNESS, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_CONTRAST, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, 128/1)

    def horizontalSlider_focus_1Function(self):
        self.current_cap = 1
        self.spinBox_focus_1.setValue(self.horizontalSlider_focus_1.value())
        self.focus_1 = self.horizontalSlider_focus_1.value() *5
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_FOCUS, self.focus_1)

    def spinBox_focus_1Function(self):
        self.horizontalSlider_focus_1.setValue(self.spinBox_focus_1.value())

    #-- CAP_PROP_BRIGHTNESS, 0~255, step=1, default=128
    def horizontalSlider_brightness_1Function(self):
        self.current_cap = 1
        self.spinBox_brightness_1.setValue(self.horizontalSlider_brightness_1.value())
        self.brightness_1 = self.horizontalSlider_brightness_1.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_1/1)

    def spinBox_brightness_1Function(self):
        self.horizontalSlider_brightness_1.setValue(
            self.spinBox_brightness_1.value())

    #-- CAP_PROP_CONTRAST, 0~255, step=1
    def horizontalSlider_contrast_1Function(self):
        self.current_cap = 1
        self.spinBox_contrast_1.setValue(self.horizontalSlider_contrast_1.value())
        self.contrast_1 = self.horizontalSlider_contrast_1.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_CONTRAST, self.contrast_1/1)

    def spinBox_contrast_1Function(self):
        self.horizontalSlider_contrast_1.setValue(self.spinBox_contrast_1.value())

    #-- CAP_PROP_SATURATION, 0~255, step=1
    def horizontalSlider_saturation_1Function(self):
        self.current_cap = 1
        self.spinBox_saturation_1.setValue(
            self.horizontalSlider_saturation_1.value())
        self.saturation_1 = self.horizontalSlider_saturation_1.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, self.saturation_1/1)

    def spinBox_saturation_1Function(self):
        self.horizontalSlider_saturation_1.setValue(
            self.spinBox_saturation_1.value())

    #-- CAP_PROP_EXPOSURE, 3~2047, step=1, default=250
    def horizontalSlider_exposure_1Function(self):
        self.current_cap = 1
        self.spinBox_exposure_1.setValue(self.horizontalSlider_exposure_1.value())
        self.exposure_1 = self.horizontalSlider_exposure_1.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_1)

    def spinBox_exposure_1Function(self):
        self.horizontalSlider_exposure_1.setValue(self.spinBox_exposure_1.value())

    #-- CAP_PROP_SHARPNESS, 0~255, step=1, default=128
    def horizontalSlider_sharpness_1Function(self):
        self.current_cap = 1
        self.spinBox_sharpness_1.setValue(
            self.horizontalSlider_sharpness_1.value())
        self.sharpness_1 = self.horizontalSlider_sharpness_1.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness_1/1)

    def spinBox_sharpness_1Function(self): 
        self.horizontalSlider_sharpness_1.setValue(
            self.spinBox_sharpness_1.value())



    @pyqtSlot(np.ndarray)
    def update_image0(self, img):
        """Updates the image_label with a new opencv image"""
        # print(len(ll))
        
        if self.canvas_mode0 != 'fiducial' and self.canvas_mode0 != 'roi':
            self.temp_canvas0 = img.copy()
            qt_img0 = self.convert_cv_qt(self.temp_canvas0,640,360)
            self.label_screen_0.setPixmap(qt_img0)
 
        elif self.canvas_mode0 == 'fiducial':
            if self.state_draw_rect:
                # cv2.rectangle(self.temp_canvas,(self.past_x*2,self.past_y*2),(self.present_x*2,self.present_y*2),(0,255,0),1)
                qt_img0 = self.convert_cv_qt(self.temp_canvas0,640,360)
                self.draw_rect(qt_img0,self.past_x,self.past_y,self.present_x,self.present_y,Qt.green)
                self.label_screen_0.setPixmap(qt_img0)
        elif self.canvas_mode0 == 'roi':
            if self.state_draw_rect:
                # cv2.rectangle(self.temp_canvas,(self.past_x*2,self.past_y*2),(self.present_x*2,self.present_y*2),(0,255,0),1)
                qt_img0 = self.convert_cv_qt(self.temp_canvas0,640,360)
                self.draw_rect(qt_img0,self.past_x,self.past_y,self.present_x,self.present_y,Qt.yellow)
                self.label_screen_0.setPixmap(qt_img0)


    @pyqtSlot(np.ndarray)
    def update_image1(self, img):
        """Updates the image_label with a new opencv image"""
        # print(len(ll))
        
        if self.canvas_mode1 != 'fiducial' and self.canvas_mode1 != 'roi':
            self.temp_canvas1 = img.copy()
            qt_img1 = self.convert_cv_qt(self.temp_canvas1,640,360)
            self.label_screen_1.setPixmap(qt_img1)

        elif self.canvas_mode1 == 'fiducial':
            if self.state_draw_rect:
                # cv2.rectangle(self.temp_canvas,(self.past_x*2,self.past_y*2),(self.present_x*2,self.present_y*2),(0,255,0),1)
                qt_img1 = self.convert_cv_qt(self.temp_canvas1,640,360)
                self.draw_rect(qt_img1,self.past_x,self.past_y,self.present_x,self.present_y,Qt.green)
                self.label_screen_1.setPixmap(qt_img1)
        elif self.canvas_mode1 == 'roi':
            if self.state_draw_rect:
                # cv2.rectangle(self.temp_canvas,(self.past_x*2,self.past_y*2),(self.present_x*2,self.present_y*2),(0,255,0),1)
                qt_img1 = self.convert_cv_qt(self.temp_canvas1,640,360)
                self.draw_rect(qt_img1,self.past_x,self.past_y,self.present_x,self.present_y,Qt.yellow)
                self.label_screen_1.setPixmap(qt_img1)
        
        
    
    def convert_cv_qt(self, cv_img, disply_width, display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    #WindowClass의 인스턴스 생성
    myWindow = WindowClass()
    #프로그램 화면을 보여주는 코드
    myWindow.show()
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()