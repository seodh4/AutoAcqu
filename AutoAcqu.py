import sys
from typing_extensions import ParamSpecKwargs
from PyQt5.QtWidgets import*
from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QEvent, QBasicTimer
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

import sip
import json
import os.path as osp

from DBHandler4 import *

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

        self.fiducial_trigger_mode = 'pass'
        

        self.cam_num = cam_num
        self.setting_ID = ''
        self.tempimg = []
        self.videowhile = False
        self.videocapture_on = False
        
        self.pre_fiducial_center = (0,0)
        self.autoannotation_flag = False

    def camon(self,cam_width):
        self.cap = cv2.VideoCapture(self.cam_num,cv2.CAP_V4L2)
        if self.cap.isOpened():
            print(self.cap,"Webcam online.")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
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
            
            # if self.videocapture_on == True:
            #     if self.videowhile == False:
            #         tm = datetime.datetime.today()
            #         datestring = str(tm.year) + str(tm.month) + str(tm.day)
            #         tmstring = str(tm.hour) +''+ str(tm.minute) +''+ str(tm.second) +'_'+ str(tm.microsecond)[1]

            #         self.createFolder(self.path+'/'+self.setting_ID+'/img/')

            #         fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            #         # out = cv2.VideoWriter(self.path+'/'+datestring+'/'+self.setting_ID+'/'+ 'data' +'/' + 'cam' + str(self.cam_num) +'_'+ datestring+'_'+tmstring + '.avi', fourcc, 25.0, (640, 480))
            #         out = cv2.VideoWriter(self.path+'/'+self.setting_ID+'/' + 'cam' + str(self.cam_num) +'_'+ datestring+'_'+tmstring + '.avi', fourcc, 25.0, (1280, 720))

            #         self.videowhile = True
            #     if self.videowhile and self.videocapture_on:
            #         out.write(self.cv_img)


            if self.fiducial_on == True:
                fiducial_center, fiducial_box, angle, check = fiducial_marker(self.cv_img, self.trainKP, self.trainDesc, self.trainImg, self.im_aspect_ratio, self.im_height, self.im_width, self.im_area)
                if check == False:
                    pass
                else:
                    

                    tm = datetime.datetime.today()
                    datestring = str(tm.year) + str('{0:02d}'.format(tm.month)) + str('{0:02d}'.format(tm.day))
                    tmstring = str('{0:02d}'.format(tm.hour)) +''+ str('{0:02d}'.format(tm.minute)) +''+ str('{0:02d}'.format(tm.second)) +'_'+ str(tm.microsecond)[1]
                    
                    imgfile_name = 'cam' + str(self.cam_num) +'_'+ datestring+'_'+tmstring + '.jpg'
                    save_path = self.path+self.setting_ID

                    if self.fiducial_trigger_mode == 'pass':
                        

                        if self.pre_fiducial_center[1] < 360 and fiducial_center[1] > 360:
                            if self.setting_ID == '':
                                    cv2.imwrite(self.path+'/'+'temp'+'/'+imgfile_name, self.cv_img)
                            else:
                                self.createFolder(save_path+'/img/')

                                cv2.imwrite(save_path+'/img/' + imgfile_name ,self.cv_img)
                                
                                if self.autoannotation_flag:
                                    jsondata = self.autoannotation(self.fiducial_img, self.master_img, self.master_fiducial_center, self.shapes, self.cv_img, save_path, imgfile_name)
                                    with open(save_path+'/img/' + imgfile_name[:-4]+'.json', "w") as f:
                                        json.dump(jsondata, f, ensure_ascii=False, indent=2)

                                    insertDB_from_json(jsondata, 1, save_path, imgfile_name)

                            cv2.polylines(self.cv_img, fiducial_box, True, (0,0,255), 5)

                        cv2.circle(self.cv_img,fiducial_center,5,(0,255,0),5)
                        cv2.circle(self.cv_img,self.pre_fiducial_center,5,(0,255,0),5)
                        cv2.line(self.cv_img, fiducial_center, self.pre_fiducial_center, (0,0,255), thickness = 2,lineType = None, shift =None)
                        cv2.line(self.cv_img, (0,360), (1280,360), (255,0,255), thickness = 2,lineType = None, shift =None)

                        cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 2)
                        self.pre_fiducial_center = fiducial_center
                        
                    elif self.fiducial_trigger_mode == 'roi':
                        polya = Polygon([(self.roi_point[0], self.roi_point[1]), (self.roi_point[0]+self.roi_point[2], self.roi_point[1]), (self.roi_point[0]+self.roi_point[2], self.roi_point[1]+self.roi_point[3]), (self.roi_point[0], self.roi_point[1]+self.roi_point[3])]) 
                        polyb = Polygon([(fiducial_box[0][0][0], fiducial_box[0][0][1]), (fiducial_box[0][1][0], fiducial_box[0][1][1]), (fiducial_box[0][2][0], fiducial_box[0][2][1]), (fiducial_box[0][3][0], fiducial_box[0][3][1])]) 

                        contain_condition=polya.contains(polyb)

                        if self.state == 'ready':
                            if contain_condition == True:
                                
                                if self.setting_ID == '':
                                    cv2.imwrite(self.path+'/'+'temp'+'/' + 'cam' + str(self.cam_num) +'_'+ datestring+'_'+tmstring + '.jpg',self.cv_img)
                                else:
                                    self.createFolder(self.path+'/'+self.setting_ID+'/img/')
                                    cv2.imwrite(self.path+'/'+self.setting_ID+'/'+ 'img' +'/' + 'cam' + str(self.cam_num) +'_'+ datestring+'_'+tmstring + '.jpg',self.cv_img)

                                cv2.polylines(self.cv_img, fiducial_box, True, (0,0,255), 5)
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
            # cv2.putText()
            s_fps = "%0.0f FPS" % fps
            cv2.putText(self.cv_img, str(s_fps), (40, 40),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            self.change_pixmap_signal.emit(self.cv_img)

        # shut down capture system 
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


    def createFolder(self, directory):
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            except OSError:
                print ('Error: Creating directory. ' +  directory) 





    def set_master_img(self,masterimg_path,masterlabel_path):

    
        with open(masterlabel_path, 'r', ) as f:
            label_dict = json.load(f)
        
        self.shapes = label_dict['shapes']
 
        # # filename=filename,
        # # shapes=shapes,
        # # imagePath=imagePath,
        # # imageData=imageData,
        # imageHeight=self.image.height(),
        # imageWidth=self.image.width(),

        self.master_img=cv2.imread(masterimg_path)

     

        for shape in self.shapes:
            if shape['label'] == 'fiducial':
                fiducial_points = shape['points']
        
    
        self.fiducial_img = self.master_img[int(fiducial_points[0][1]):int(fiducial_points[1][1]), int(fiducial_points[0][0]):int(fiducial_points[1][0])]

        trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, imCrop, im_area = search_feature(self.fiducial_img)
        self.master_fiducial_center, fiducial_box, angle, check = fiducial_marker(self.master_img, trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width,im_area)

        if check:
            print('Good')
            print(self.shapes)
        else:
            print('False')
        

    def IoU(self, box1, box2):
        # box = (x1, y1, x2, y2)
        # box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box1_area = (box1[1][0] - box1[0][0] + 1) * (box1[1][1] - box1[0][1] + 1)

        # box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        box2_area = (box2[1][0] - box2[0][0] + 1) * (box2[1][1] - box2[0][1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0][0], box2[0][0])
        y1 = max(box1[0][1], box2[0][1])
        x2 = min(box1[1][0], box2[1][0])
        y2 = min(box1[1][1], box2[1][1])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou
        

    def autoannotation(self, fiducial_img, master_img, master_fiducial_center, shapes, img, savepath, imagefilename):


        fiducial_w = fiducial_img.shape[1]
        fiducial_h = fiducial_img.shape[0]

        # img=cv2.imread(image_path)
        trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, imCrop, im_area = search_feature(fiducial_img)
        fiducial_center, fiducial_box, angle, check = fiducial_marker(img, trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width,im_area)

        if check:
            # print(fiducial_center)
            
            d_x = fiducial_center[0] - master_fiducial_center[0]
            d_y = fiducial_center[1] - master_fiducial_center[1]


            fiducial_center
            con1 = fiducial_center[0] - int(fiducial_w/2) > -1
            con2 = fiducial_center[0] + int(fiducial_w/2) < img.shape[1]+1
            con3 = fiducial_center[1] - int(fiducial_h/2) > -1
            con4 = fiducial_center[1] + int(fiducial_h/2) < img.shape[0]+1
            
            fiducial_box_point = [(fiducial_center[0] - int(fiducial_w/2), fiducial_center[1] - int(fiducial_h/2)) , (fiducial_center[0] + int(fiducial_w/2), fiducial_center[1] + int(fiducial_h/2))]

            # print(fiducial_box)
            # con1 = fiducial_box[0][0][0] > -1 and fiducial_box[0][0][1] > -1
            # con2 = fiducial_box[0][2][0] < img.shape[1]+1 and fiducial_box[0][2][1] < img.shape[0]+1


            if con1 and con2 and con3 and con4:

                updata_shapes=[]
                for shape in shapes:
                    label = shape["label"]

                    if label != 'fiducial':
                    
                        points = shape["points"]

                        updata_points = []
                        for point in points:
                            # angle = angle * -1
                            # angle_x, angle_y = self.get_angle(fiducial_center[0], fiducial_center[1],point[0], point[1], round(angle))

                            # updata_point_x = angle_x + d_x
                            # updata_point_y = angle_y + d_y

                            updata_point_x = point[0] + d_x
                            updata_point_y = point[1] + d_y

                            updata_points.append((int(updata_point_x),int(updata_point_y)))
                    
                        
                        if updata_points[1][0] < img.shape[1] and updata_points[1][1] < img.shape[0]:
                            updata_points_h = int(updata_points[1][1]) - int(updata_points[0][1])
                            updata_points_w = int(updata_points[1][0]) - int(updata_points[0][0])

                            updata_points_h = int(updata_points_h / 2)
                            updata_points_w = int(updata_points_w / 2)

                            search_img_x1 = int(updata_points[0][0]) - updata_points_w if (int(updata_points[0][0]) - updata_points_w) > 0 else 0
                            search_img_x2 = int(updata_points[1][0]) + updata_points_w if (int(updata_points[1][0]) + updata_points_w) < img.shape[1] else img.shape[1]

                            search_img_y1 = int(updata_points[0][1]) - updata_points_h if (int(updata_points[0][1]) - updata_points_h) > 0 else 0
                            search_img_y2 = int(updata_points[1][1]) + updata_points_h if (int(updata_points[1][1]) + updata_points_h) < img.shape[0] else img.shape[0]
                            
                            # print(updata_points[0][0],updata_points[1][0],updata_points[0][1],updata_points[1][1])
                            # print(updata_points_w, updata_points_w)
                            # print(search_img_x1,search_img_x2,search_img_y1,search_img_y2)
                            
                            search_img=img[search_img_y1:search_img_y2, search_img_x1:search_img_x2]
                            src = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)

                            
                            feature_img=master_img[int(points[0][1]):int(points[1][1]), int(points[0][0]):int(points[1][0])]
                            feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)
                            
                            
                            result = cv2.matchTemplate(src, feature_img, cv2.TM_CCOEFF_NORMED)
                            
                            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                            x, y = maxLoc
                            h, w = feature_img.shape

                            new_point_x1 = search_img_x1 + x
                            new_point_y1 = search_img_y1 + y

                            new_point_x2 = search_img_x1 + x +  w
                            new_point_y2 = search_img_y1 + y + h

                            new_point = [(new_point_x1,new_point_y1), (new_point_x2,new_point_y2)]

                            iou=self.IoU(new_point, updata_points)



                            img = cv2.rectangle(img, (new_point[0]), (new_point[1]) , (0, 0, 255), 1)
                            cv2.putText(img, str(iou), (new_point[0]),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                            


                            # h, w = feature_img.shape
                            # result = cv2.matchTemplate(src, feature_img, cv2.TM_CCOEFF_NORMED)

                            # threshold = 0.4
                            # loc = np.where(result >= threshold)

                            # candidate_new_point = []
                            # candidate_new_point_iou = []

                            # for pt in zip(*loc[::-1]):
                                
                                
                            #     x = pt[0]
                            #     y = pt[1]

                            #     new_point_x1 = search_img_x1 + x
                            #     new_point_y1 = search_img_y1 + y

                            #     new_point_x2 = search_img_x1 + x +  w
                            #     new_point_y2 = search_img_y1 + y + h

                            #     new_point = [(int(new_point_x1),int(new_point_y1)), (int(new_point_x2),int(new_point_y2))]

                            #     iou=self.IoU(new_point, updata_points)
                            #     candidate_new_point.append(new_point)
                            #     candidate_new_point_iou.append(iou)
                            
                            # if len(candidate_new_point_iou) == 0:
                            #     n_new_point = updata_points
                            # else:
                            #     n_new_point = candidate_new_point[candidate_new_point_iou.index(max(candidate_new_point_iou))]


                            # n_new_point

                            # cv2.rectangle(img, (n_new_point[0]), (n_new_point[1]) , (0, 0, 255), 1)
                            
                            # try:
                            #     cv2.putText(img, str(max(candidate_new_point_iou)), (n_new_point[0]),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),1)
                            # except:
                            #     pass


                            shape_type = shape["shape_type"]
                            flags = shape["flags"]
                            group_id = shape["group_id"]

                            
                            img = cv2.rectangle(img, (search_img_x1, search_img_y1), (search_img_x2, search_img_y2) , (0, 255, 255), 1)


                            updata_shape = dict(
                                        label=label,
                                        points=new_point,
                                        group_id=group_id,
                                        shape_type=shape_type,
                                        flags=flags,)
                            updata_shapes.append(updata_shape)
                        

                data = dict(
                version='4.5.13',
                flags={},
                shapes=updata_shapes,
                imagePath=imagefilename,
                imageData=None,
                imageHeight=img.shape[0],
                imageWidth=img.shape[1],)

                # print(savepath, imagefilename)
                # print(savepath + imagefilename[:-4]+'.json')
                
                    # print(self.currentPath()+'/'+label_file)
                
                cv2.circle(img, (master_fiducial_center[0], master_fiducial_center[1]), 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.line(img, (master_fiducial_center[0], master_fiducial_center[1]), (fiducial_center[0], fiducial_center[1]), (0, 0, 255), thickness=2, lineType=None, shift=None)
                cv2.circle(img, (fiducial_center[0], fiducial_center[1]), 5, (0, 255, 0), -1, cv2.LINE_AA)

                img = cv2.rectangle(img, (fiducial_box_point[0][0], fiducial_box_point[0][1]), (fiducial_box_point[1][0], fiducial_box_point[1][1]) , (0, 255, 255), 2)
                img = cv2.rectangle(img, (fiducial_box[0][0]), (fiducial_box[0][2]) , (255, 0, 0), 2)
                
                # cv2.imwrite(savepath+imagefilename[-4]+'_aaa'+'.jpg',img)

                return data








class save_list_widget(QWidget):

    get_ID_signal = pyqtSignal(str)

    def __init__(self,img,ID):
        super().__init__()

        if len(img) == 2:
            self.qt_img1 = img[0]
            self.qt_img2 = img[1]
        else:
            self.qt_img2 = img[0]

        self.ID = ID

        self.initUI()
        

    
    def initUI(self):
        self.setAcceptDrops(True)

        self.image_widget_QVBox = QVBoxLayout()
        self.image_widget_namelabel = QLabel()
        self.image_widget_namelabel.setText(self.ID)


        self.image_widget_Hbox = QHBoxLayout()
        self.image_widget_label = QLabel()
        self.image_widget_label2 = QLabel()
        self.image_widget_label.setPixmap(self.qt_img1)
        self.image_widget_Hbox.addWidget(self.image_widget_label)
        self.image_widget_label2.setPixmap(self.qt_img2)
        self.image_widget_Hbox.addWidget(self.image_widget_label2)

        self.image_widget_QVBox.addWidget(self.image_widget_namelabel)
        self.image_widget_QVBox.addLayout(self.image_widget_Hbox)

        # self.img_btn = QPushButton()
        # self.img_btn.setEnabled(False)
        # self.img_btn.setText('Drag Here!') 
        # self.image_widget_layout.addWidget(self.img_btn)
        self.setLayout(self.image_widget_QVBox)

    def mousePressEvent(self, event):
        # self.setStyleSheet('background:yellow')
        # print(self.ID)
        self.get_ID_signal.emit(self.ID)



form_class = uic.loadUiType("./acq.ui")[0]

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

        self.select_id = None
        self.setting_ID_list = []

        self.save_fiducial_point = [[0,0,0,0],[0,0,0,0]]
        self.save_roi_point = [[0,0,0,0],[0,0,0,0]]
        self.save_fiducial_img = [0] * 2
        self.save_ISP = [0] * 2

        zroimg = np.zeros((512,512,3), np.uint8)
        self.save_fiducial_img[0] = zroimg
        self.save_fiducial_img[1] = zroimg

       

        self.path = './data/'

        self.timer = QBasicTimer()
        self.timer.start(1000, self)

        self.pushButton_videocapture.clicked.connect(self.pushButton_videocaptureFunction)
        self.videocapture_on = False
        self.pushButton_save.clicked.connect(self.pushButton_saveFunction)
        self.pushButton_load.clicked.connect(self.pushButton_loadFunction)
        self.pushButton_del.clicked.connect(self.pushButton_delFunction)

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

        self.pushButton_fiducial_change.clicked.connect(self.pushButton_fiducial_changeFunction)

        self.lineEdit_path.textChanged.connect(self.lineEdit_pathFunction)
        self.lineEdit_ID.textChanged.connect(self.lineEdit_IDFunction)

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

        self.radioButton_roi_0.clicked.connect(self.groupboxRad_trigger_Function)
        self.radioButton_pass_0.clicked.connect(self.groupboxRad_trigger_Function)
        self.radioButton_roi_1.clicked.connect(self.groupboxRad_trigger_Function)
        self.radioButton_pass_1.clicked.connect(self.groupboxRad_trigger_Function)
     
        self.checkBox_autoannotation_0.stateChanged.connect(self.checkBox_autoannotation_0_Function)
        self.pushButton_path_masterimg_0.clicked.connect(self.pushButton_path_masterimg_0_Function)
        self.checkBox_autoannotation_1.stateChanged.connect(self.checkBox_autoannotation_1_Function)
        self.pushButton_path_masterimg_1.clicked.connect(self.pushButton_path_masterimg_1_Function)

        widget = QWidget()
        self.layout = QVBoxLayout()
        widget.setLayout(self.layout)
        self.scrollArea.setWidget(widget)
        self.screen_reverse=False
        

        self.thread = []

        ret=glob.glob('/dev/video*')

        for video in ret:
            numbers = re.findall("\d+", video)
            self.thread.append(VideoThread(int(numbers[0])))
            # print(numbers)


        # read setting.json

        self.load_setting_list()

                

        if len(self.thread) == 1:
            self.thread[0].change_pixmap_signal.connect(self.update_image0)
            # crop !!!
            self.thread[0].imcrop_qt_img = self.convert_cv_qt(self.save_fiducial_img[0],120,120)
            self.label_screen_fiducial_0.setPixmap(self.thread[0].imcrop_qt_img)
            self.label_screen_fiducial_1.setPixmap(self.thread[0].imcrop_qt_img)

        elif len(self.thread) == 2:
            self.thread[0].change_pixmap_signal.connect(self.update_image0)
            self.thread[0].imcrop_qt_img = self.convert_cv_qt(self.save_fiducial_img[0],120,120)
            self.thread[1].change_pixmap_signal.connect(self.update_image1)
            self.thread[1].imcrop_qt_img = self.convert_cv_qt(self.save_fiducial_img[1],120,120)
            self.label_screen_fiducial_0.setPixmap(self.thread[0].imcrop_qt_img)
            self.label_screen_fiducial_1.setPixmap(self.thread[1].imcrop_qt_img)
        
    

       
        # start the thread
        
        self.label_screen_0.mousePressEvent = self.screen_0_mousePressEvent
        self.label_screen_0.mouseMoveEvent = self.screen_0_mouseMoveEvent
        self.label_screen_0.mouseReleaseEvent = self.screen_0_mouseReleaseEvent

        self.label_screen_1.mousePressEvent = self.screen_1_mousePressEvent
        self.label_screen_1.mouseMoveEvent = self.screen_1_mouseMoveEvent
        self.label_screen_1.mouseReleaseEvent = self.screen_1_mouseReleaseEvent

        self.width = 1280

        
    def pushButton_videocaptureFunction(self):
        if self.videocapture_on == False:
            self.pushButton_videocapture.setText("&STOP")
            for th in self.thread:
                th.videocapture_on = True
            self.videocapture_on = True

        else:
            self.pushButton_videocapture.setText("&record")
            for th in self.thread:
                th.videocapture_on = False
                th.videowhile = False
            self.videocapture_on = False

    
    def timerEvent(self, e):
        self.diskLabel = './'
        total, used, free = shutil.disk_usage(self.diskLabel)

        total = total/ 1000000
        used = used / 1000000
        free = free / 1000000
        str_total = "%0.0f MB" % total
        str_used = "%0.0f MB" % used
        str_free = "%0.0f MB" % free

        step= (used/total) * 100
        self.progressBar_disk.setValue(step)
        self.label_disk.setText(str_free +' '+ str_used +' '+ str_total)
    


    def load_setting_list(self):
        
        self.setting_ID_list = [] 
        with open('./setting/setting.json', 'r') as f:
            self.json_data = json.load(f)
            for setting_ID in self.json_data:
                # print(setting_ID)
                self.setting_ID_list.append(setting_ID)

                fiducial_img_0_path = self.json_data[setting_ID]['fiducial_img_0']
                fiducial_img_1_path = self.json_data[setting_ID]['fiducial_img_1']
                fiducial_img_0 = cv2.imread(fiducial_img_0_path)
                fiducial_img_1 = cv2.imread(fiducial_img_1_path)
                qt1=self.convert_cv_qt(fiducial_img_0,120,120)
                qt2=self.convert_cv_qt(fiducial_img_1,120,120)

                self.save_list_widget = save_list_widget([qt1,qt2],str(setting_ID))
                self.layout.addWidget(self.save_list_widget)
                self.save_list_widget.get_ID_signal.connect(self.get_ID_signalFunction)



    def lineEdit_IDFunction(self):

        for th in self.thread:
            th.setting_ID = self.lineEdit_ID.text()



    def pushButton_fiducial_changeFunction(self):

        self.thread.reverse()
        if self.screen_reverse == False:
            self.label_screen_0.setGeometry(670,210,640,360)
            self.label_screen_1.setGeometry(10,210,640,360)
            self.screen_reverse = True
        else:
            self.label_screen_0.setGeometry(10,210,640,360)
            self.label_screen_1.setGeometry(670,210,640,360)
            self.screen_reverse = False
        


    def pushButton_saveFunction(self):
        
        ID = self.lineEdit_ID.text()

        ISP_0 = []
        ISP_1 = []
        if len(self.thread) == 1:
            zroimg = np.zeros((512,512,3), np.uint8)
            zroqt = self.convert_cv_qt(zroimg,120,120)
            self.save_list_widget = save_list_widget([self.thread[0].imcrop_qt_img,zroqt],ID)

            self.focus_0 = int(self.horizontalSlider_focus_0.value())
            self.brightness_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_BRIGHTNESS))
            self.contrast_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_CONTRAST))
            self.saturation_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_SATURATION))
            self.exposure_0 = int(self.horizontalSlider_exposure_0.value())
            self.sharpness_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_SHARPNESS))

            ISP_0 = [self.focus_0, self.brightness_0, self.contrast_0, self.saturation_0, self.exposure_0, self.sharpness_0]


        elif len(self.thread) == 2:
            self.save_list_widget = save_list_widget([self.thread[0].imcrop_qt_img,self.thread[1].imcrop_qt_img],ID)

            self.focus_0 = int(self.horizontalSlider_focus_0.value())
            self.brightness_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_BRIGHTNESS))
            self.contrast_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_CONTRAST))
            self.saturation_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_SATURATION))
            self.exposure_0 = int(self.horizontalSlider_exposure_0.value())
            self.sharpness_0 = int(self.thread[0].cap.get(cv2.CAP_PROP_SHARPNESS))

            self.focus_1 = int(self.horizontalSlider_focus_1.value())
            self.brightness_1 = int(self.thread[1].cap.get(cv2.CAP_PROP_BRIGHTNESS))
            self.contrast_1 = int(self.thread[1].cap.get(cv2.CAP_PROP_CONTRAST))
            self.saturation_1 = int(self.thread[1].cap.get(cv2.CAP_PROP_SATURATION))
            self.exposure_1 = int(self.horizontalSlider_exposure_1.value())
            self.sharpness_1 = int(self.thread[1].cap.get(cv2.CAP_PROP_SHARPNESS))

            ISP_0 = [self.focus_0, self.brightness_0, self.contrast_0, self.saturation_0, self.exposure_0, self.sharpness_0]
            ISP_1 = [self.focus_1, self.brightness_1, self.contrast_1, self.saturation_1, self.exposure_1, self.sharpness_1]

        setting_json = {
            ID : {
                'ISP_0' : ISP_0,
                'ISP_1' : ISP_1,
                'fiducial_point': self.save_fiducial_point,
                'roi_point': self.save_roi_point,
                # 'fiducial_img_0': self.path+'/'+ID+'/setting/'+ID+'_0.jpg',
                # 'fiducial_img_1': self.path+'/'+ID+'/setting/'+ID+'_1.jpg',
                'fiducial_img_0': './setting/img/'+ID+'_0.jpg',
                'fiducial_img_1': './setting/img/'+ID+'_1.jpg',
            }
        }

        self.json_data.update(setting_json)



        with open('./setting/setting.json', 'w', encoding='utf-8') as make_file:
            json.dump(self.json_data, make_file, indent="\t")

        cv2.imwrite('./setting/img/'+ID+'_0.jpg', self.save_fiducial_img[0])
        cv2.imwrite('./setting/img/'+ID+'_1.jpg', self.save_fiducial_img[1])
        

        
        for idx in range(self.layout.count()):
            self.layout.itemAt(idx).widget().deleteLater()

        


        save_json = setting_json

        tm = datetime.datetime.today()
        datestring = str(tm.year) + str('{0:02d}'.format(tm.month)) + str('{0:02d}'.format(tm.day))
        
        self.createFolder(self.path+'/'+ID+'/setting/')
        
        with open(self.path+'/'+ID+'/setting/'+ ID + '.json', 'w', encoding='utf-8') as make_file:
            json.dump(save_json, make_file, indent="\t")
    

        cv2.imwrite(self.path+'/'+ID+'/setting/'+ID+'_0.jpg', self.save_fiducial_img[0])
        cv2.imwrite(self.path+'/'+ID+'/setting/'+ID+'_1.jpg', self.save_fiducial_img[1])


        self.load_setting_list()
        # self.save_list.append()
        # for th in self.thread:
        #     print(th.imcrop_qt_img)

        #     img=cv2.imread('default.jpg')
        #     self.image_widget = image_widget(img,img,0)
        #     layout.addWidget(self.image_widget)

        

    def pushButton_loadFunction(self):
        self.load_setting(self.select_id)
        for idx in range(self.layout.count()):
            self.layout.itemAt(idx).widget().deleteLater()
        self.load_setting_list()


    def pushButton_delFunction(self):
        del(self.json_data[self.select_id])
        with open('./setting/setting.json', 'w', encoding='utf-8') as make_file:
            json.dump(self.json_data, make_file, indent="\t")
        # self.layout.itemAt(idx).widget().deleteLater()
        for idx in range(self.layout.count()):
            self.layout.itemAt(idx).widget().deleteLater()
        self.load_setting_list()
        os.remove('./setting/img/'+self.select_id+'_0.jpg')
        os.remove('./setting/img/'+self.select_id+'_1.jpg')

      


    @pyqtSlot(str)
    def get_ID_signalFunction(self,a):
        print(a)
        self.select_id = a
        
        for idx in range(self.layout.count()):
            self.layout.itemAt(idx).widget().setStyleSheet("""
        QWidget {
        
        }
        """)

        idx = self.setting_ID_list.index(a)
        print(idx)
        self.layout.itemAt(idx).widget().setStyleSheet("""border:2px solid rgb(0, 0, 0);""")
        

            
    def load_setting(self, ID):
        
        
        
        self.lineEdit_ID.setText(ID)
        

        self.save_fiducial_point[0] = self.json_data[ID]['fiducial_point'][0]
        self.save_fiducial_point[1] = self.json_data[ID]['fiducial_point'][1]
        self.save_roi_point[0] = self.json_data[ID]['roi_point'][0]
        self.save_roi_point[1] = self.json_data[ID]['roi_point'][1]

        
    
        self.save_ISP[0] = self.json_data[ID]['ISP_0']
        self.save_ISP[1] = self.json_data[ID]['ISP_1']
        self.save_fiducial_img[0] = cv2.imread(self.json_data[ID]['fiducial_img_0'])
        self.save_fiducial_img[1] = cv2.imread(self.json_data[ID]['fiducial_img_1'])



        self.spinBox_focus_0.setValue(int(self.save_ISP[0][0]))
        self.horizontalSlider_focus_0.setValue(int(self.save_ISP[0][0]))
        self.spinBox_brightness_0.setValue(int(self.save_ISP[0][1]))
        self.horizontalSlider_brightness_0.setValue(int(self.save_ISP[0][1]))
        self.spinBox_contrast_0.setValue(int(self.save_ISP[0][2]))   
        self.horizontalSlider_contrast_0.setValue(int(self.save_ISP[0][2]))
        self.spinBox_saturation_0.setValue(int(self.save_ISP[0][3]))    
        self.horizontalSlider_saturation_0.setValue(int(self.save_ISP[0][3]))
        self.spinBox_exposure_0.setValue(int(self.save_ISP[0][4]))     
        self.horizontalSlider_exposure_0.setValue(int(self.save_ISP[0][4]))
        self.spinBox_sharpness_0.setValue(int(self.save_ISP[0][5]))    
        self.horizontalSlider_sharpness_0.setValue(int(self.save_ISP[0][5]))

        self.spinBox_focus_1.setValue(int(self.save_ISP[1][0]))
        self.horizontalSlider_focus_1.setValue(int(self.save_ISP[1][0]))
        self.spinBox_brightness_1.setValue(int(self.save_ISP[1][1]))
        self.horizontalSlider_brightness_1.setValue(int(self.save_ISP[1][1]))
        self.spinBox_contrast_1.setValue(int(self.save_ISP[1][2]))   
        self.horizontalSlider_contrast_1.setValue(int(self.save_ISP[1][2]))
        self.spinBox_saturation_1.setValue(int(self.save_ISP[1][3]))    
        self.horizontalSlider_saturation_1.setValue(int(self.save_ISP[1][3]))
        self.spinBox_exposure_1.setValue(int(self.save_ISP[1][4]))     
        self.horizontalSlider_exposure_1.setValue(int(self.save_ISP[1][4]))
        self.spinBox_sharpness_1.setValue(int(self.save_ISP[1][5]))    
        self.horizontalSlider_sharpness_1.setValue(int(self.save_ISP[1][5]))
        
        

        if len(self.thread) == 1:
            self.thread[0].imcrop_qt_img=self.convert_cv_qt(self.save_fiducial_img[0],120,120)
            self.label_screen_fiducial_0.setPixmap(self.thread[0].imcrop_qt_img)
            self.load_fiducial_0(self.thread[0],self.save_fiducial_img[1])

            self.thread[0].cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.thread[0].cap.set(cv2.CAP_PROP_FOCUS, self.save_ISP[0][0]*5)
            self.thread[0].cap.set(cv2.CAP_PROP_BRIGHTNESS, self.save_ISP[0][1]/1)
            self.thread[0].cap.set(cv2.CAP_PROP_CONTRAST, self.save_ISP[0][2]/1)
            self.thread[0].cap.set(cv2.CAP_PROP_SATURATION, self.save_ISP[0][3]/1)
            self.thread[0].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.thread[0].cap.set(cv2.CAP_PROP_EXPOSURE, self.save_ISP[0][4])
            self.thread[0].cap.set(cv2.CAP_PROP_SHARPNESS, self.save_ISP[0][5]/1)
            self.thread[0].roi_point = self.save_roi_point[0]
            
        elif len(self.thread) == 2:
            self.thread[0].imcrop_qt_img=self.convert_cv_qt(self.save_fiducial_img[0],120,120)
            self.thread[1].imcrop_qt_img=self.convert_cv_qt(self.save_fiducial_img[1],120,120)
            self.label_screen_fiducial_0.setPixmap(self.thread[0].imcrop_qt_img)
            self.label_screen_fiducial_1.setPixmap(self.thread[1].imcrop_qt_img)
            self.load_fiducial_0(self.thread[0],self.save_fiducial_img[0])
            self.load_fiducial_1(self.thread[1],self.save_fiducial_img[1])

            self.thread[0].cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.thread[0].cap.set(cv2.CAP_PROP_FOCUS, self.save_ISP[0][0]*5)
            self.thread[0].cap.set(cv2.CAP_PROP_BRIGHTNESS, self.save_ISP[0][1]/1)
            self.thread[0].cap.set(cv2.CAP_PROP_CONTRAST, self.save_ISP[0][2]/1)
            self.thread[0].cap.set(cv2.CAP_PROP_SATURATION, self.save_ISP[0][3]/1)
            self.thread[0].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.thread[0].cap.set(cv2.CAP_PROP_EXPOSURE, self.save_ISP[0][4])
            self.thread[0].cap.set(cv2.CAP_PROP_SHARPNESS, self.save_ISP[0][5]/1)
            self.thread[0].roi_point = self.save_roi_point[0]

            self.thread[1].cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.thread[1].cap.set(cv2.CAP_PROP_FOCUS, self.save_ISP[1][0]*5)
            self.thread[1].cap.set(cv2.CAP_PROP_BRIGHTNESS, self.save_ISP[1][1]/1)
            self.thread[1].cap.set(cv2.CAP_PROP_CONTRAST, self.save_ISP[1][2]/1)
            self.thread[1].cap.set(cv2.CAP_PROP_SATURATION, self.save_ISP[1][3]/1)
            self.thread[1].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.thread[1].cap.set(cv2.CAP_PROP_EXPOSURE, self.save_ISP[1][4])
            self.thread[1].cap.set(cv2.CAP_PROP_SHARPNESS, self.save_ISP[1][5]/1)
            self.thread[1].roi_point = self.save_roi_point[1]
        
        # save_json = {ID : self.json_data[ID]}

        # tm = datetime.datetime.today()
        # datestring = str(tm.year) + str('{0:02d}'.format(tm.month)) + str('{0:02d}'.format(tm.day))

        # if os.path.exists(self.path+'/'+ID+'/'):
        #     pass
        # else:
        #     os.makedirs(self.path+'/'+ID+'/')
        
        # with open(self.path+'/'+ID+'/setting/'+ ID + '.json', 'w', encoding='utf-8') as make_file:
        #     json.dump(save_json, make_file, indent="\t")
        
        # if os.path.exists(self.path+'/'+ID+'/'+ 'setting'+'/'):
        #     pass
        # else:
        #     os.makedirs(self.path+'/'+ID+'/'+ 'setting'+'/')
        # cv2.imwrite(self.path+'/'+ID+'/'+ 'setting'+'/'+ID+'_0.jpg', self.save_fiducial_img[0])
        # cv2.imwrite(self.path+'/'+ID+'/'+ 'setting'+'/'+ID+'_1.jpg', self.save_fiducial_img[1])


    def groupboxRadFunction(self) :
        if self.radioButton_1280.isChecked() : self.width = 1280
        elif self.radioButton_960.isChecked() : self.width = 960

    def groupboxRad_trigger_Function(self) :

        if len(self.thread) == 2:
            if self.radioButton_roi_0.isChecked() : self.thread[0].fiducial_trigger_mode = 'roi'
            elif self.radioButton_pass_0.isChecked() : self.thread[0].fiducial_trigger_mode = 'pass'

            if self.radioButton_roi_1.isChecked() : self.thread[1].fiducial_trigger_mode = 'roi'
            elif self.radioButton_pass_1.isChecked() : self.thread[1].fiducial_trigger_mode = 'pass'
        else:
            if self.radioButton_roi_0.isChecked() : self.thread[0].fiducial_trigger_mode = 'roi'
            elif self.radioButton_pass_0.isChecked() : self.thread[0].fiducial_trigger_mode = 'pass'


    

    def pushButton_camonFunction(self):
        for thread in self.thread:
            thread.camon(self.width)
            thread.start()
            # self.clithread.start()

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
            tmstring = str('{0:02d}'.format(tm.hour)) +''+ str('{0:02d}'.format(tm.minute)) +''+ str('{0:02d}'.format(tm.second)) +'_'+ str(tm.microsecond)[1]

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
                self.save_roi_point[0] = self.thread[self.current_cap].roi_point

        self.present_x = event.pos().x()
        self.present_y = event.pos().y()
        # print(self.past_x,self.past_y,self.present_x,self.present_y)
        self.past_x = 0
        self.past_y = 0
        self.present_x = 0
        self.present_y = 0


    def load_fiducial_0(self, thread, crop):
        # self.lineEdit.setText(str(a))

        imCrop=crop
        thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width, thread.imCrop, thread.im_area = search_feature(imCrop)


       

    def fiducial_signal_0(self, thread):

        # print(thread.fiducial_point)
        
        imCrop=crop_img(thread.tempimg,thread.fiducial_point)
        try:
            thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width, thread.imCrop, thread.im_area = search_feature(imCrop)

            fiducial_center, fiducial_box, angle, check = fiducial_marker(thread.cv_img, thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width,thread.im_area)
            print(thread.imCrop.shape)
            thread.imcrop_qt_img = self.convert_cv_qt(thread.imCrop,120,120) 
            self.label_screen_fiducial_0.setPixmap(thread.imcrop_qt_img)

            if check:
                thread.fiducial_center_offset = fiducial_center
                thread.fiducial_angle_offset = angle
                print('good')
                # print(thread.fiducial_point)
                self.save_fiducial_point[0] = thread.fiducial_point
                self.save_fiducial_img[0] = thread.imCrop

            else:
                print('again')
        except:
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
                self.save_roi_point[1] = self.thread[self.current_cap].roi_point

        self.present_x = event.pos().x()
        self.present_y = event.pos().y()
        # print(self.past_x,self.past_y,self.present_x,self.present_y)
        self.past_x = 0
        self.past_y = 0
        self.present_x = 0
        self.present_y = 0


    def load_fiducial_1(self, thread, crop):

        imCrop=crop
        thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width, thread.imCrop, thread.im_area = search_feature(imCrop)

       

    def fiducial_signal_1(self, thread):

        # print(thread.fiducial_point)
        imCrop=crop_img(thread.tempimg,thread.fiducial_point)
        try:
            thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width, thread.imCrop, thread.im_area = search_feature(imCrop)
            fiducial_center, fiducial_box, angle, check = fiducial_marker(thread.cv_img, thread.trainKP, thread.trainDesc, thread.trainImg, thread.im_aspect_ratio, thread.im_height, thread.im_width,thread.im_area)
            # print(thread.imCrop.shape)
            thread.imcrop_qt_img = self.convert_cv_qt(thread.imCrop,120,120) 
            self.label_screen_fiducial_1.setPixmap(thread.imcrop_qt_img)

            if check:
                thread.fiducial_center_offset = fiducial_center
                thread.fiducial_angle_offset = angle
                print('good')
                # print(thread.fiducial_point)
                self.save_fiducial_point[1] = thread.fiducial_point
                self.save_fiducial_img[1] = thread.imCrop


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
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_BRIGHTNESS, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_CONTRAST, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, 128/1)


        
        self.focus_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_FOCUS))
        p_focus_0 = 0
        self.brightness_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_BRIGHTNESS))
        self.contrast_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_CONTRAST))
        self.saturation_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SATURATION))
        self.exposure_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_EXPOSURE))
        p_exposure_0 = 0
        self.sharpness_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SHARPNESS))

        
        while p_focus_0 != self.focus_0 or p_exposure_0 != self.exposure_0:
            p_focus_0 = self.focus_0
            p_exposure_0 = self.exposure_0
            time.sleep(1)
            self.focus_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_FOCUS))
            self.exposure_0 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_EXPOSURE))
        


        print('focus : ' + str(self.focus_0))
        print('brightness : ' + str(self.brightness_0))
        print('contrast : ' + str(self.contrast_0))
        print('saturation : ' + str(self.saturation_0))
        print('exposure : ' + str(self.exposure_0))
        print('sharpness : ' + str(self.sharpness_0))


        self.spinBox_focus_0.setValue(int(self.focus_0/5))
        self.horizontalSlider_focus_0.setValue(int(self.focus_0/5))
        self.spinBox_brightness_0.setValue(int(self.brightness_0))
        self.horizontalSlider_brightness_0.setValue(int(self.brightness_0))
        self.spinBox_contrast_0.setValue(int(self.contrast_0))   
        self.horizontalSlider_contrast_0.setValue(int(self.contrast_0))
        self.spinBox_saturation_0.setValue(int(self.saturation_0))    
        self.horizontalSlider_saturation_0.setValue(int(self.saturation_0))
        self.spinBox_exposure_0.setValue(int(self.exposure_0))     
        self.horizontalSlider_exposure_0.setValue(int(self.exposure_0))
        self.spinBox_sharpness_0.setValue(int(self.sharpness_0))    
        self.horizontalSlider_sharpness_0.setValue(int(self.sharpness_0))
        


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
        self.horizontalSlider_brightness_0.setValue(self.spinBox_brightness_0.value())

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
        self.spinBox_saturation_0.setValue(self.horizontalSlider_saturation_0.value())
        self.saturation_0 = self.horizontalSlider_saturation_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, self.saturation_0/1)

    def spinBox_saturation_0Function(self):
        self.horizontalSlider_saturation_0.setValue(self.spinBox_saturation_0.value())

    #-- CAP_PROP_EXPOSURE, 3~2047, step=1, default=250
    def horizontalSlider_exposure_0Function(self):
        self.current_cap = 0
        self.spinBox_exposure_0.setValue(self.horizontalSlider_exposure_0.value())
        self.exposure_0 = self.horizontalSlider_exposure_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_0)

    def spinBox_exposure_0Function(self):
        self.horizontalSlider_exposure_0.setValue(self.spinBox_exposure_0.value())

    #-- CAP_PROP_SHARPNESS, 0~255, step=1, default=128
    def horizontalSlider_sharpness_0Function(self):
        self.current_cap = 0
        self.spinBox_sharpness_0.setValue(self.horizontalSlider_sharpness_0.value())
        self.sharpness_0 = self.horizontalSlider_sharpness_0.value()
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness_0/1)

    def spinBox_sharpness_0Function(self): 
        self.horizontalSlider_sharpness_0.setValue(self.spinBox_sharpness_0.value())




    def pushButton_autoip_1Function(self):
        self.current_cap = 1
        
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        # 
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  

        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_BRIGHTNESS, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_CONTRAST, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SATURATION, 128/1)
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_SHARPNESS, 128/1)



        self.focus_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_FOCUS))
        p_focus_1 = 1
        self.brightness_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_BRIGHTNESS))
        self.contrast_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_CONTRAST))
        self.saturation_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SATURATION))
        self.exposure_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_EXPOSURE))
        p_exposure_1 = 1
        self.sharpness_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_SHARPNESS))


        while p_focus_1 != self.focus_1 or p_exposure_1 != self.exposure_1:
            p_focus_1 = self.focus_1
            p_exposure_1 = self.exposure_1
            time.sleep(1)
            self.focus_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_FOCUS))
            self.exposure_1 = int(self.thread[self.current_cap].cap.get(cv2.CAP_PROP_EXPOSURE))
            



        print('focus : ' + str(self.focus_1))
        print('brightness : ' + str(self.brightness_1))
        print('contrast : ' + str(self.contrast_1))
        print('saturation : ' + str(self.saturation_1))
        print('exposure : ' + str(self.exposure_1))
        print('sharpness : ' + str(self.sharpness_1))


        self.spinBox_focus_1.setValue(int(self.focus_1/5))
        self.horizontalSlider_focus_1.setValue(int(self.focus_1/5))
        self.spinBox_brightness_1.setValue(int(self.brightness_1))
        self.horizontalSlider_brightness_1.setValue(int(self.brightness_1))
        self.spinBox_contrast_1.setValue(int(self.contrast_1))   
        self.horizontalSlider_contrast_1.setValue(int(self.contrast_1))
        self.spinBox_saturation_1.setValue(int(self.saturation_1))    
        self.horizontalSlider_saturation_1.setValue(int(self.saturation_1))
        self.spinBox_exposure_1.setValue(int(self.exposure_1))     
        self.horizontalSlider_exposure_1.setValue(int(self.exposure_1))
        self.spinBox_sharpness_1.setValue(int(self.sharpness_1))    
        self.horizontalSlider_sharpness_1.setValue(int(self.sharpness_1))


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
        self.thread[self.current_cap].cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
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
            self.thread[0].tempimg = self.temp_canvas0
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
            self.thread[1].tempimg = self.temp_canvas1
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



    def pushButton_path_masterimg_0_Function(self):
        temp = QFileDialog.getOpenFileName(self, "select path","", "Json Files(*.json)")
        self.thread[0].masterlabel_path = temp[0]
        self.lineEdit_masterimg_path_0.setText(self.thread[0].masterlabel_path)
        self.thread[0].masterimg_path = self.thread[0].masterlabel_path[:-5] + ".jpg"
    


    def checkBox_autoannotation_0_Function(self,state):
        if state == Qt.Checked:
            self.thread[0].set_master_img(self.thread[0].masterimg_path, self.thread[0].masterlabel_path)
            self.thread[0].autoannotation_flag = True
        else:
            self.thread[0].autoannotation_flag = False
          

    def pushButton_path_masterimg_1_Function(self):
        temp = QFileDialog.getOpenFileName(self, "select path","", 'Json Files(*.json)')
        self.thread[1].masterlabel_path = temp[0]
        self.lineEdit_masterimg_path_1.setText(self.thread[1].masterlabel_path)
        self.thread[1].masterimg_path = self.thread[1].masterlabel_path[:-5] + ".jpg"
    


    def checkBox_autoannotation_1_Function(self,state):
        if state == Qt.Checked:
            self.thread[1].set_master_img(self.thread[1].masterimg_path, self.thread[1].masterlabel_path)
            self.thread[1].autoannotation_flag = True
        else:
            self.thread[1].autoannotation_flag = False
        






    


 


    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory) 


if __name__ == "__main__":

    app = QApplication(sys.argv)
 
    myWindow = WindowClass()

 
    myWindow.show()
   



    app.exec_()
