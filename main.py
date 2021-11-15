import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from plots_cv_platenumber import *
from TrackingAPI8_0915 import *   
from tqdm import tqdm

from utils.plots import colors, plot_one_box

import cv2
import numpy as np
import time
from time import sleep
from detect import *
from canvas import Canvas

#Thread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(tuple) #np.ndarray
    plate_info_signal = pyqtSignal(list)
    video_info_signal = pyqtSignal(tuple)
    test_info_signal = pyqtSignal(np.ndarray)
    right_data_signal = pyqtSignal(np.ndarray)


    
    def __init__(self):
        super().__init__()

        # para
        self.inf_fps = 0
        self._run_flag = True
        self.mode = 'cam'
        self.inference_mode = 1
        self.sw_inference = False
        self.view_img = False
        self.conf = 0.4
        self.iounms = 0.2
        self.oneinference = False
        self.savefile = ''

        self.roi = (0,0,0,0)
        self.fiducial_center_offset= (0,0)
        self.fiducial_angle_offset = 0

        self.canvas_mode = 'default'

        self.right_datas = []

        # yolo
        self.yolo_PNR = yolov5('PNR_210902.pt')
        self.yolo_PCB = yolov5('best.pt')
        self.yolo_PCBOCR = yolov5('PCBOCR_210915.pt')
        self.yolo_OCR = yolov5('OCR.pt')
        self.infersize = 960
        
        # video
        self.framenum = 0
        self.length = 0
        self.video_ctr = 'play'


        self.disply_width = 1280
        self.display_height = 720


    def get_iou(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou


    def get_angle(self,x,y, angle):
        rad = angle * (float)(math.pi / 180.0)
        nx = int(math.cos(rad)*x - math.sin(rad)*y)
        ny = int(math.sin(rad)*x + math.cos(rad)*y)
        return nx, ny



    def get_perspect_roi(self, fiducial_center, roi, fiducial_center_offset, angle):

        fiducial_center_x = fiducial_center[0]
        fiducial_center_y = fiducial_center[1]

        fiducial_center_x_differ = fiducial_center_x - fiducial_center_offset[0]
        fiducial_center_y_differ = fiducial_center_y - fiducial_center_offset[1]

        parallel_roi_x1 = roi[0] + fiducial_center_x_differ - fiducial_center_x
        parallel_roi_y1 = roi[1] + fiducial_center_y_differ - fiducial_center_y
        parallel_roi_x2 = roi[0] + roi[2] + fiducial_center_x_differ - fiducial_center_x
        parallel_roi_y2 = roi[1] + fiducial_center_y_differ - fiducial_center_y
        parallel_roi_x3 = roi[0] + roi[2] + fiducial_center_x_differ - fiducial_center_x
        parallel_roi_y3 = roi[1] + roi[3] + fiducial_center_y_differ - fiducial_center_y
        parallel_roi_x4 = roi[0] + fiducial_center_x_differ - fiducial_center_x
        parallel_roi_y4 = roi[1] + roi[3] + fiducial_center_y_differ - fiducial_center_y

        Perspect_roi_x1, Perspect_roi_y1 = self.get_angle(parallel_roi_x1, parallel_roi_y1, round(angle))
        Perspect_roi_x2, Perspect_roi_y2 = self.get_angle(parallel_roi_x2, parallel_roi_y2, round(angle))
        Perspect_roi_x3, Perspect_roi_y3 = self.get_angle(parallel_roi_x3, parallel_roi_y3, round(angle))
        Perspect_roi_x4, Perspect_roi_y4 = self.get_angle(parallel_roi_x4, parallel_roi_y4, round(angle))
        
        Perspect_roi_x1 = Perspect_roi_x1 + fiducial_center_x
        Perspect_roi_y1 = Perspect_roi_y1 + fiducial_center_y
        Perspect_roi_x2 = Perspect_roi_x2 + fiducial_center_x
        Perspect_roi_y2 = Perspect_roi_y2 + fiducial_center_y
        Perspect_roi_x3 = Perspect_roi_x3 + fiducial_center_x
        Perspect_roi_y3 = Perspect_roi_y3 + fiducial_center_y
        Perspect_roi_x4 = Perspect_roi_x4 + fiducial_center_x
        Perspect_roi_y4 = Perspect_roi_y4 + fiducial_center_y


        Perspect_roi_x1 = int(int(Perspect_roi_x1 / 4) * 4)
        Perspect_roi_x2 = int(int(Perspect_roi_x2 / 4) * 4)
        Perspect_roi_x3 = int(int(Perspect_roi_x3 / 4) * 4)
        Perspect_roi_x4 = int(int(Perspect_roi_x4 / 4) * 4)
        Perspect_roi_y1 = int(int(Perspect_roi_y1 / 4) * 4)
        Perspect_roi_y2 = int(int(Perspect_roi_y2 / 4) * 4)
        Perspect_roi_y3 = int(int(Perspect_roi_y3 / 4) * 4)
        Perspect_roi_y4 = int(int(Perspect_roi_y4 / 4) * 4)
        
        Perspect_roi = (Perspect_roi_x1, Perspect_roi_y1, Perspect_roi_x2, Perspect_roi_y2, Perspect_roi_x3, Perspect_roi_y3, Perspect_roi_x4, Perspect_roi_y4)

        return Perspect_roi


    def draw_box(self, xyxy, angle, Perspect_roi_x1, Perspect_roi_y1):

        res=[0]*8

        parallel_res_x1 = xyxy[0]
        parallel_res_y1 = xyxy[1]
        parallel_res_x2 = xyxy[2]
        parallel_res_y2 = xyxy[1]
        parallel_res_x3 = xyxy[2]
        parallel_res_y3 = xyxy[3]
        parallel_res_x4 = xyxy[0]
        parallel_res_y4 = xyxy[3]

        Perspect_res_x1, Perspect_res_y1 = self.get_angle(parallel_res_x1, parallel_res_y1, round(angle))
        Perspect_res_x2, Perspect_res_y2 = self.get_angle(parallel_res_x2, parallel_res_y2, round(angle))
        Perspect_res_x3, Perspect_res_y3 = self.get_angle(parallel_res_x3, parallel_res_y3, round(angle))
        Perspect_res_x4, Perspect_res_y4 = self.get_angle(parallel_res_x4, parallel_res_y4, round(angle))
    
        res[0] = Perspect_res_x1 + Perspect_roi_x1
        res[1] = Perspect_res_y1 + Perspect_roi_y1
        res[2] = Perspect_res_x2 + Perspect_roi_x1
        res[3] = Perspect_res_y2 + Perspect_roi_y1
        res[4] = Perspect_res_x3 + Perspect_roi_x1
        res[5] = Perspect_res_y3 + Perspect_roi_y1
        res[6] = Perspect_res_x4 + Perspect_roi_x1
        res[7] = Perspect_res_y4 + Perspect_roi_y1

        return res


    def run(self):
        PTime = time.time()

        if self.mode == 'image':
            pass


        if self.mode == 'video':
            self.cap = cv2.VideoCapture('./samples/test.mp4')
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            vedio_info = self.length, fps
            self.video_info_signal.emit(vedio_info)
            # print(self.length,fps)
            self.framenum = 0
            # for framenum in range(0, length):
            while self.framenum < self.length and self._run_flag:
                # print(self.framenum)

                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.framenum)
                ret, self.cv_img = self.cap.read()
                if ret and self.sw_inference:
                    pred, imgshape, self.inf_fps = self.yolo_PNR.detect(self.cv_img, self.conf, self.iounms)
                    boxs = []
                    for i, det in enumerate(pred):
                        if len(det):
                            det[:, :4] = scale_coords(imgshape[2:], det[:, :4], self.cv_img.shape).round()
                            for *xyxy, conf, cls in reversed(det):
                            
                                if self.view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = self.yolo_PNR.names[c]
                                    self.cv_img = plot_one_box(xyxy, self.cv_img, label=label, color=(0,128,255),txt_color=(0, 0, 0), line_width=1)

                                box = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                                boxs.append([box, conf, cls])
                    
  

                    if self.video_ctr == 'play':
                        if self.length > self.framenum + 1:
                            self.framenum = self.framenum + 1
                    plate_type, points_list, im = plot_word_box(boxs, self.cv_img, self.yolo_PNR.names, line_thickness=1)
                    self.change_pixmap_signal.emit((self.cv_img, fps, self.inf_fps))   

                    if len(plate_type) == 0 or len(points_list) == 0 or plate_type[0][0] == '' or plate_type[0][1] == '' or plate_type[0][2] == '' or len(plate_type) > 8 or len(plate_type[0]) > 3:
                        if self.video_ctr == 'play':
                            if self.length > self.framenum + 1:
                                self.framenum = self.framenum + 1
                    else:
                        plate_info = plate_parse(plate_type, points_list, self.cv_img)
                        self.plate_info_signal.emit(plate_info)
                        if self.video_ctr == 'play':
                            if self.length > self.framenum + 1:
                                self.framenum = self.framenum + 1
                else:
                    self.change_pixmap_signal.emit((self.cv_img, fps, self.inf_fps))
                    # sleep(1/fps*2)
                    if self.video_ctr == 'play':
                        self.framenum = self.framenum + 1
                if ret is False:
                    break

        if self.mode == 'cam':

            pre_box = []
            pre_angle = 0
            pre_fiducial_center = (0,0)
    
            # self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            self.cap = cv2.VideoCapture(0)


            self.cap.set(3, self.disply_width)
            self.cap.set(4, self.display_height)


            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(str(fps)+' '+str(width)+' '+ str(height))

            fiducial_center_x_differ = 0
            fiducial_center_y_differ = 0
            fiducial_angle_differ = 0
            angle = 0

            fiducial_center_x = 0
            fiducial_center_y = 0
            

            Perspect_roi_x1 = self.roi[0]
            Perspect_roi_y1 = self.roi[1]
            Perspect_roi_x2 = self.roi[0]+self.roi[2]
            Perspect_roi_y2 = self.roi[1]
            Perspect_roi_x3 = self.roi[0]+self.roi[2]
            Perspect_roi_y3 = self.roi[1]+ self.roi[3]
            Perspect_roi_x4 = self.roi[0]
            Perspect_roi_y4 = self.roi[1]+self.roi[3]

            while self.mode == 'cam':
                if self._run_flag:
                    ret, self.cv_img = self.cap.read()


                    if self.sw_inference == False and self.canvas_mode != 'ROI':
                        # pts_roi = np.array([[self.roi[0], self.roi[1]], [self.roi[0]+self.roi[2], self.roi[1]], [self.roi[0]+self.roi[2], self.roi[1]+self.roi[3]], [self.roi[0], self.roi[1]+self.roi[3]]])
                        pts_roi = np.array([[Perspect_roi_x1, Perspect_roi_y1], [Perspect_roi_x2, Perspect_roi_y2], [Perspect_roi_x3, Perspect_roi_y3], [Perspect_roi_x4, Perspect_roi_y4]])

                        cv2.polylines(self.cv_img, [pts_roi], True, (255, 0, 255), 1)
                    

                    # self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
                    if ret and self.sw_inference:
                        # PCB
                        if self.inference_mode == 1:

                            fiducial_center, fiducial_box, angle, check = fiducial_marker(self.cv_img, self.trainKP, self.trainDesc, self.trainImg, self.im_aspect_ratio, self.im_height, self.im_width, self.im_area)
                            if check == False:
                                fiducial_box = pre_box
                                angle = pre_angle
                                fiducial_center = pre_fiducial_center

                            Perspect_roi = self.get_perspect_roi(fiducial_center, self.roi, self.fiducial_center_offset, angle)

                            (Perspect_roi_x1, Perspect_roi_y1, Perspect_roi_x2, Perspect_roi_y2, Perspect_roi_x3, Perspect_roi_y3, Perspect_roi_x4, Perspect_roi_y4) = Perspect_roi

                            pre_box = fiducial_box
                            pre_angle = angle
                            pre_fiducial_center = fiducial_center

                            pts_roi = np.array([[Perspect_roi_x1, Perspect_roi_y1], [Perspect_roi_x2, Perspect_roi_y2], [Perspect_roi_x3, Perspect_roi_y3], [Perspect_roi_x4, Perspect_roi_y4]])
                            cv2.polylines(self.cv_img, [pts_roi], True, (255, 0, 255), 1)

                            spts = np.array([[Perspect_roi_x1,Perspect_roi_y1],[Perspect_roi_x2,Perspect_roi_y2],[Perspect_roi_x3,Perspect_roi_y3],[Perspect_roi_x4,Perspect_roi_y4]], np.float32)
                            dstPoint = np.array([[self.roi[0],self.roi[1]], [self.roi[0] + self.roi[2],self.roi[1]], [self.roi[0] + self.roi[2],self.roi[1] + self.roi[3]], [self.roi[0],self.roi[1] + self.roi[3]]], np.float32)
                            matrix = cv2.getPerspectiveTransform(spts, dstPoint)
                            dst = cv2.warpPerspective(self.cv_img, matrix, (self.cv_img.shape[1],self.cv_img.shape[0]))
                            ddd=dst[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]] 
                            # self.test_info_signal.emit(ddd)
                            # print(ddd.shape)
                            # crop_roi = self.cv_img[parallel_roi_y1:parallel_roi_y2, parallel_roi_x1:parallel_roi_x2]
                            # print(angle , fiducial_center)
                            ddd = np.ascontiguousarray(ddd)
                            if ddd.shape[0] % 32 == 0 and ddd.shape[0] > 31 and ddd.shape[1] % 32 == 0 and ddd.shape[1] > 31:
                                
                                # self.infersize = ddd.shape[1]
                                pred, imgshape, self.inf_fps = self.yolo_PCB.detect(ddd, self.conf, self.iounms)
                                       
                                boxs = []
                                # gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                for i, det in enumerate(pred):
                                    if len(det):
                                        #-- Rescale boxes from img_size to im0 size
                                        det[:, :4] = scale_coords(imgshape[2:], det[:, :4], ddd.shape).round()
                                        for *xyxy, conf, cls in reversed(det):
                                            
                                            c = int(cls)  # integer class
                                            label = self.yolo_PCB.names[c]
                                            
                                            if self.view_img:  # Add bbox to image
                                                # c = int(cls)  # integer class
                                                # label = self.yolo_PCB.names[c]
                                       
                                                ddd = plot_one_box(xyxy, ddd, label=label, color=colors(c, True),txt_color=(0, 0, 0), line_width=1)
                                                res = [0] * 8
                                                res = self.draw_box(xyxy, angle, Perspect_roi_x1, Perspect_roi_y1)
                                                self.cv_img = plot_one_box(res, self.cv_img, label=label, color=colors(c, True),txt_color=(0, 0, 0), line_width=1, poly = True)


                                            # temp = []
                                            # for right_data in self.right_datas:
                                            #     right_cord = right_data[0]
                                            #     iou=self.get_iou(right_cord,xyxy)
                                            #     if iou > 0.2:
                                            #         temp.append([right_data,iou])
                                            # temp.sort(key = lambda x:x[1], reverse=True)

                                            # if len(temp) > 0:
                                            #     if cls == temp[0][0][2]:
                                            #         print(temp[0][0])
                                                

                                            box = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                                            boxs.append([box, conf, cls])

                                # gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                                # self.draw_box(pred, self.cv_img, imgshape, angle, Perspect_roi)

                                    

                                # for box in boxs:
                                #     temp = [] 
                                #     for right_data in self.right_datas:
                                #         right_cord = right_data[0]
                                #         box_cord = box[0]
                                #         iou=self.get_iou(right_cord,box_cord)
                                #         if iou > 0.5:
                                #             temp.append([right_data,box_cord])
                                #     temp.sort(key = lambda x:x[1], reverse=True)

                                #     res = self.draw_box(box_cord, angle, Perspect_roi_x1, Perspect_roi_y1)
                                #     pts_res = np.array([[res[0], res[1]], [res[2], res[3]], [res[4], res[5]], [res[6], res[7]]])

                                #     if len(temp) > 0:
                                #         if box[2] == temp[0][0][2]: # 맞는거
                                #             cv2.polylines(self.cv_img, [pts_res], True, (0, 255, 0), 2)
                                #             # print(box,temp[0][0])
                                #         else:
                                #             cv2.polylines(self.cv_img, [pts_res], True, (0, 0, 255), 2)
                                #     else:
                                #         cv2.polylines(self.cv_img, [pts_res], True, (255, 0, 0), 2)

                                shapes = np.zeros_like(self.cv_img, np.uint8)

                                for right_data in self.right_datas:
                                    temp = [] 
                                    for box in boxs:
                                        right_cord = right_data[0]
                                        box_cord = box[0]
                                        iou=self.get_iou(right_cord,box_cord)
                                        if iou > 0.2:
                                            temp.append([box,iou])
                                    temp.sort(key = lambda x:x[1], reverse=True)

                                    res = self.draw_box(right_cord, angle, Perspect_roi_x1, Perspect_roi_y1)
                                    pts_res = np.array([[res[0], res[1]], [res[2], res[3]], [res[4], res[5]], [res[6], res[7]]])

                                    if len(temp) > 0:
                                        if right_data[2] == temp[0][0][2]: # 맞는거
                                            # cv2.polylines(self.cv_img, [pts_res], True, (0, 255, 0), 2)
                                            cv2.fillConvexPoly(shapes, pts_res, (50, 255, 50))
                                            # print(box,temp[0][0])
                                        else:
                                            # cv2.polylines(self.cv_img, [pts_res], True, (0, 0, 255), 2)
                                            cv2.fillConvexPoly(shapes, pts_res, (50, 50, 255))
                                    else:  
                                        # cv2.polylines(self.cv_img, [pts_res], True, (255, 0, 0), 2)
                                        cv2.fillConvexPoly(shapes,pts_res, (255, 50, 50))
                                
                                out = self.cv_img.copy()
                                alpha = 0.5
                                mask = shapes.astype(bool)
                                out[mask] = cv2.addWeighted(self.cv_img, alpha, shapes, 1 - alpha, 0)[mask]  

                                # for r in result:
                                #     print(r)
                            
                                # for right_data in self.right_datas:
                                #     # print(right_data) n  oi_y1)
                                    
                                #     # pts_right_data = np.array([[right_data[0][0], right_data[0][1]], [right_data[0][2], right_data[0][1]], [right_data[0][2], right_data[0][3]], [right_data[0][0], right_data[0][3]]])
                                #     right_data_cord = self.draw_box(xyxy, angle, Perspect_roi_x1, Perspect_roi_y1)
                                #     pts_right_data_cord = np.array([[right_data_cord[0], right_data_cord[1]], [right_data_cord[2], right_data_cord[3]], [right_data_cord[4], right_data_cord[5]], [right_data_cord[6], right_data_cord[7]]])

                                #     cv2.polylines(self.cv_img, [pts_right_data_cord], True, (255, 0, 255), 1)
                                

                            self.test_info_signal.emit(ddd)
                            cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 2)



                        # 데이터셋 만들때
                        # OCR
                        if self.inference_mode == 2:
                            
                            fiducial_center, fiducial_box, angle, check = fiducial_marker(self.cv_img, self.trainKP, self.trainDesc, self.trainImg, self.im_aspect_ratio, self.im_height, self.im_width, self.im_area)
                            
                            # print(fiducial_center, fiducial_box, angle, check)
                            if check == False:
                                fiducial_box = pre_box
                                angle = pre_angle
                                fiducial_center = pre_fiducial_center


                            Perspect_roi = self.get_perspect_roi(fiducial_center, self.roi, self.fiducial_center_offset, angle)

                            (Perspect_roi_x1, Perspect_roi_y1, Perspect_roi_x2, Perspect_roi_y2, Perspect_roi_x3, Perspect_roi_y3, Perspect_roi_x4, Perspect_roi_y4) = Perspect_roi


                            pre_box = fiducial_box
                            pre_angle = angle
                            pre_fiducial_center = fiducial_center

                            pts_roi = np.array([[Perspect_roi_x1, Perspect_roi_y1], [Perspect_roi_x2, Perspect_roi_y2], [Perspect_roi_x3, Perspect_roi_y3], [Perspect_roi_x4, Perspect_roi_y4]])
                            cv2.polylines(self.cv_img, [pts_roi], True, (255, 0, 255), 1)

                            spts = np.array([[Perspect_roi_x1,Perspect_roi_y1],[Perspect_roi_x2,Perspect_roi_y2],[Perspect_roi_x3,Perspect_roi_y3],[Perspect_roi_x4,Perspect_roi_y4]], np.float32)
                            dstPoint = np.array([[self.roi[0],self.roi[1]], [self.roi[0] + self.roi[2],self.roi[1]], [self.roi[0] + self.roi[2],self.roi[1] + self.roi[3]], [self.roi[0],self.roi[1] + self.roi[3]]], np.float32)
                            matrix = cv2.getPerspectiveTransform(spts, dstPoint)
                            dst = cv2.warpPerspective(self.cv_img, matrix, (self.cv_img.shape[1],self.cv_img.shape[0]))
                            ddd=dst[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]] 
                      
                            ddd = np.ascontiguousarray(ddd)
                            if ddd.shape[0] % 32 == 0 and ddd.shape[0] > 31 and ddd.shape[1] % 32 == 0 and ddd.shape[1] > 31:
                                
                                # self.infersize = ddd.shape[1]
                                pred, imgshape, self.inf_fps = self.yolo_OCR.detect(ddd, self.conf, self.iounms)
                                
                               
                                boxs = []
                                # gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                for i, det in enumerate(pred):
                                    if len(det):
                                        #-- Rescale boxes from img_size to im0 size
                                        det[:, :4] = scale_coords(imgshape[2:], det[:, :4], ddd.shape).round()
                                        for *xyxy, conf, cls in reversed(det):
                                            if self.view_img:  # Add bbox to image
                                                c = int(cls)  # integer class
                                                # label = str(c)
                                                label = self.yolo_OCR.names[c] 
                                                ddd = plot_one_box(xyxy, ddd, label=label, color=(0,128,255),txt_color=(0, 0, 0), line_width=1)
                                                res = [0] * 8
                                                res = self.draw_box(xyxy, angle, Perspect_roi_x1, Perspect_roi_y1)
                                                self.cv_img = plot_one_box(res, self.cv_img, label=label, color=(0,128,255),txt_color=(0, 0, 0), line_width=1, poly = True)

                                            
                                            box = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                                            boxs.append([box, conf, cls])

                            self.test_info_signal.emit(ddd)

                            cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 2)


                        #PCB OCR
                        if self.inference_mode == 10:
                            
                            fiducial_center, fiducial_box, angle, check = fiducial_marker(self.cv_img, self.trainKP, self.trainDesc, self.trainImg, self.im_aspect_ratio, self.im_height, self.im_width, self.im_area)
                            if check == False:
                                fiducial_box = pre_box
                                angle = pre_angle
                                fiducial_center = pre_fiducial_center
                            
                            # fiducial_center_x = int(int(fiducial_center[0] / 4) * 4)
                            # fiducial_center_y = int(int(fiducial_center[1] / 4) * 4)

                            Perspect_roi = self.get_perspect_roi(fiducial_center, self.roi, self.fiducial_center_offset, angle)

                            (Perspect_roi_x1, Perspect_roi_y1, Perspect_roi_x2, Perspect_roi_y2, Perspect_roi_x3, Perspect_roi_y3, Perspect_roi_x4, Perspect_roi_y4) = Perspect_roi


                            pre_box = fiducial_box
                            pre_angle = angle
                            pre_fiducial_center = fiducial_center

                            pts_roi = np.array([[Perspect_roi_x1, Perspect_roi_y1], [Perspect_roi_x2, Perspect_roi_y2], [Perspect_roi_x3, Perspect_roi_y3], [Perspect_roi_x4, Perspect_roi_y4]])
                            cv2.polylines(self.cv_img, [pts_roi], True, (255, 0, 255), 1)

                            spts = np.array([[Perspect_roi_x1,Perspect_roi_y1],[Perspect_roi_x2,Perspect_roi_y2],[Perspect_roi_x3,Perspect_roi_y3],[Perspect_roi_x4,Perspect_roi_y4]], np.float32)
                            dstPoint = np.array([[self.roi[0],self.roi[1]], [self.roi[0] + self.roi[2],self.roi[1]], [self.roi[0] + self.roi[2],self.roi[1] + self.roi[3]], [self.roi[0],self.roi[1] + self.roi[3]]], np.float32)
                            matrix = cv2.getPerspectiveTransform(spts, dstPoint)
                            dst = cv2.warpPerspective(self.cv_img, matrix, (self.cv_img.shape[1],self.cv_img.shape[0]))
                            ddd=dst[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]] 
                            # self.test_info_signal.emit(ddd)
                            # print(ddd.shape)
                            # crop_roi = self.cv_img[parallel_roi_y1:parallel_roi_y2, parallel_roi_x1:parallel_roi_x2]
                            # print(angle , fiducial_center)
                            ddd = np.ascontiguousarray(ddd)
                            if ddd.shape[0] % 32 == 0 and ddd.shape[0] > 31 and ddd.shape[1] % 32 == 0 and ddd.shape[1] > 31:
                                
                                # self.infersize = ddd.shape[1]
                                pred, imgshape, self.inf_fps = self.yolo_PCBOCR.detect(ddd, self.conf, self.iounms)
                                
                                boxs = []
                                # gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                for i, det in enumerate(pred):
                                    if len(det):
                                        #-- Rescale boxes from img_size to im0 size
                                        det[:, :4] = scale_coords(imgshape[2:], det[:, :4], ddd.shape).round()
                                        for *xyxy, conf, cls in reversed(det):
                                            if self.view_img:  # Add bbox to image
                                                c = int(cls)  # integer class
                                                # label = str(c)
                                                label = self.yolo_PCBOCR.names[c]
                                                ddd = plot_one_box(xyxy, ddd, label=label, color=(0,128,255),txt_color=(0, 0, 0), line_width=1)
                                                res = [0] * 8
                                                res = self.draw_box(xyxy, angle, Perspect_roi_x1, Perspect_roi_y1)
                                                self.cv_img = plot_one_box(res, self.cv_img, label=label, color=(0,128,255),txt_color=(0, 0, 0), line_width=1, poly = True)

                                            box = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                                            boxs.append([box, conf, cls])

                            self.test_info_signal.emit(ddd)
                            # print(fiducial_box,fiducial_box2,angle)
                            # if check == False:
                            #     fiducial_box = pre_box
                            #     angle = pre_angle

                            cv2.polylines(self.cv_img, fiducial_box, True, (0,255,0), 2)

                            # cv2.drawContours(self.cv_img, fiducial_box, -1, (0, 255, 0), 2)

                            # cv2.polylines(self.cv_img, fiducial_box2, True, (0,255,0), 2)
                            # print(fiducial_box, angle)

                        # OCR
                        if self.inference_mode == 3: 

                            pred, imgshape, self.inf_fps = self.yolo_PCB.detect(self.cv_img, self.conf, self.iounms)

                            for i, det in enumerate(pred):
                                    if len(det):
                                        #-- Rescale boxes from img_size to im0 size
                                        det[:, :4] = scale_coords(imgshape[2:], det[:, :4], self.cv_img.shape).round()
                                        for *xyxy, conf, cls in reversed(det):
                                            if self.view_img:  # Add bbox to image
                                                c = int(cls)  # integer class
                                                # label = str(c)
                                                label = self.yolo_PCB.names[c]
                                                # im0s = plot_one_box(xyxy, im0s, label=label, color=colors(c, True), line_width=1)
                                                self.cv_img = plot_one_box(xyxy, self.cv_img, label=label, color=colors(c, True),txt_color=(0, 0, 0), line_width=1)

                            # 데이터 취득
                            if self.oneinference == True:
                                save_txt = True
                                ret, self.cv_img = self.cap.read()
                                print(self.savefile)

                                pred, imgshape, self.inf_fps = self.yolo_PCB.detect(self.cv_img, self.conf, self.iounms)
                                print(len(pred[0]))
                                if len(pred[0]) < 27:
                               
                                    boxs = []
                                    gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                    for i, det in enumerate(pred):
                                        if len(det):
                                            #-- Rescale boxes from img_size to im0 size
                                            det[:, :4] = scale_coords(imgshape[2:], det[:, :4], self.cv_img.shape).round()
                                            for *xyxy, conf, cls in reversed(det):
                                                if save_txt:  # Write to file
                                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                                    save_conf = False
                                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                                    with open(self.savefile + '.txt', 'a') as f:
                                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                                    cv2.imwrite(self.savefile + '.jpg', self.cv_img)
                                                

                                self.oneinference = False
                        
                        # PNR
                        if self.inference_mode == 0:
                            pred, imgshape, self.inf_fps = self.yolo_PNR.detect(self.cv_img, self.conf, self.iounms)
                            # print(pred)
                            boxs = []
                            # gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            for i, det in enumerate(pred):
                                if len(det):
                                    #-- Rescale boxes from img_size to im0 size
                                    det[:, :4] = scale_coords(imgshape[2:], det[:, :4], self.cv_img.shape).round()
                                    for *xyxy, conf, cls in reversed(det):
                                        # if save_txt:  # Write to file
                                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                        #     save_conf = False
                                        #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                        #     with open(txt_path + '.txt', 'a') as f:
                                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                        #     cv2.imwrite(txt_path + '.jpg', self.cv_img)
                                        if self.view_img:  # Add bbox to image
                                            c = int(cls)  # integer class
                                            # label = str(c)
                                            label = self.yolo_PNR.names[c]
                                            # im0s = plot_one_box(xyxy, im0s, label=label, color=colors(c, True), line_width=1)
                                            self.cv_img = plot_one_box(xyxy, self.cv_img, label=label, color=(0,128,255),txt_color=(0, 0, 0), line_width=1)

                                        box = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                                        boxs.append([box, conf, cls])
                            
                            plate_type, points_list, im = plot_word_box(boxs, self.cv_img, self.yolo_PNR.names, line_thickness=1)
                            
                            if len(plate_type) == 0 or len(points_list) == 0 or plate_type[0][0] == '' or plate_type[0][1] == '' or plate_type[0][2] == '' or len(plate_type) > 8 or len(plate_type[0]) > 3:
                                # print("not exists objects")
                                pass
                            else:
                                # print(points_list)
                                plate_info = plate_parse(plate_type, points_list, self.cv_img)
                                self.plate_info_signal.emit(plate_info)
                        # print(points_list)
                        cTime = time.time()
                        sec = cTime - PTime
                        PTime = time.time()
                        fps = 1 / (sec)
                        # s_fps = "%0.0f FPS" % fps
                        # cv2.putText(self.im0s, str(fps), (20, 40),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        if self.inference_mode == 1:
                            self.change_pixmap_signal.emit((out, fps, self.inf_fps))
                        else:
                            self.change_pixmap_signal.emit((self.cv_img, fps, self.inf_fps))
                    elif ret and self.sw_inference == False:
                        cTime = time.time()
                        sec = cTime - PTime
                        PTime = time.time()
                        fps = 1 / (sec)
                        # s_fps = "%0.0f FPS" % fps
                        # cv2.putText(self.cv_img, str(fps), (20, 40),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        self.change_pixmap_signal.emit((self.cv_img, fps, self.inf_fps))    
            #-- shut down capture system
            self.cap.release()
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self.mode = 'end'
        self._run_flag = False
        self.quit()
        self.wait()

    def save_rightdata(self):

        self.canvas_mode = 'default'
        ddd=self.cv_img[self.roi[1]:self.roi[1]+self.roi[3],self.roi[0]:self.roi[0]+self.roi[2]] 
        # self.test_info_signal.emit(ddd)
        # print(ddd.shape)
        # crop_roi = self.cv_img[parallel_roi_y1:parallel_roi_y2, parallel_roi_x1:parallel_roi_x2]
        # print(angle , fiducial_center)
        ddd = np.ascontiguousarray(ddd)

        if ddd.shape[0] % 32 == 0 and ddd.shape[0] > 31 and ddd.shape[1] % 32 == 0 and ddd.shape[1] > 31:
    
            # self.infersize = ddd.shape[1]
            pred, imgshape, fps = self.yolo_PCB.detect(ddd, self.conf, self.iounms)
            
            print(self.infersize, imgshape[2:], ddd.shape)
            self.right_datas = []
            # gn = torch.tensor(self.cv_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            for i, det in enumerate(pred):
                if len(det):
                    #-- Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(imgshape[2:], det[:, :4], ddd.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = self.yolo_PCB.names[c]
                        ddd = plot_one_box(xyxy, ddd, label=label, color=colors(c, True),txt_color=(0, 0, 0), line_width=1)
                        
                        box = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                        self.right_datas.append([box, conf, cls])
            self.right_data_signal.emit(ddd)
       

    def capture(self,savefile,savefilenum):
        
        # save = './data/' + savefile + '_'+savefilenum



        # cv2.imwrite(save +'.jpg', self.cv_img)

        br = [64, 128, 192]
        ct = [-6, -5, -4]
        st = [64, 128, 192]
        se = [64, 128, 192]

        for cti in tqdm(ct):
        
            PTime = time.time()

            self.cap.set(cv2.CAP_PROP_EXPOSURE, cti/1)
            time.sleep(0.15)
            save = './data/' + savefile + '_'+savefilenum + '_' + 'ex' + str(cti)
            cv2.imwrite(save +'.jpg', self.cv_img)
            cTime = time.time()
            sec = cTime - PTime

            print(sec)
            # for cti in ct:
            #     self.cap.set(cv2.CAP_PROP_CONTRAST, cti/1)
            #     for sti in st:
            #         self.cap.set(cv2.CAP_PROP_SATURATION, sti/1)
            #         for spi in sp:
            #             self.cap.set(cv2.CAP_PROP_SHARPNESS, spi/1)
            #             time.sleep(0.25)
            #             save = './data/' + savefile + '_'+savefilenum + '_' + 'br' + str(bri) + 'ct' + str(cti) + 'st' + str(sti) + 'sp' + str(spi)
            #             self.change_pixmap_signal.emit((self.cv_img, 0))
            #             cv2.imwrite(save +'.jpg', self.cv_img)

    





























#UI파일 연결
form_class = uic.loadUiType("form.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.record = False
        # self.screensize = (640,480) 
        self.y =0

        self.trigger_ready = False

        self.emptybox = np.full((50,150,3), (136, 138, 133), dtype=np.uint8)
        self.pushButton_capture.clicked.connect(self.pushButton_captureFunction)

        self.canvas.fiducial_signal.connect(self.fiducial_signal)
        self.canvas.roi_signal.connect(self.roi_signal)
        # self.lineEdit_infersize.textChanged.connect(self.lineEdit_infersizeFunction)

        self.pushButton_camon.clicked.connect(self.pushButton_camonFunction)
        self.pushButton_camoff.clicked.connect(self.pushButton_camoffFunction)
        self.pushButton_autoip.clicked.connect(self.pushButton_autoipFunction)
        self.pushButton_sizeup.clicked.connect(self.canvas.zoomIn)
        self.pushButton_sizedown.clicked.connect(self.canvas.zoomOut)

        self.pushButton_trigger.clicked.connect(self.pushButton_triggerFunction)
        self.pushButton_fiducial.clicked.connect(self.pushButton_fiducialFunction)
        self.pushButton_ROI.clicked.connect(self.pushButton_ROIFunction)

        self.pushButton_fiducial_ocr.clicked.connect(self.pushButton_fiducial_ocrFunction)
        self.pushButton_ROI_ocr.clicked.connect(self.pushButton_ROI_ocrFunction)

        #
        self.horizontalSlider_focus.valueChanged.connect(self.horizontalSlider_focusFunction)
        self.spinBox_focus.valueChanged.connect(self.spinBox_focusFunction)
        self.horizontalSlider_exposure.valueChanged.connect(self.horizontalSlider_exposureFunction)
        self.spinBox_exposure.valueChanged.connect(self.spinBox_exposureFunction)
        self.horizontalSlider_brightness.valueChanged.connect(self.horizontalSlider_brightnessFunction)
        self.spinBox_brightness.valueChanged.connect(self.spinBox_brightnessFunction)
        self.horizontalSlider_contrast.valueChanged.connect(self.horizontalSlider_contrastFunction)
        self.spinBox_contrast.valueChanged.connect(self.spinBox_contrastFunction)
        self.horizontalSlider_saturation.valueChanged.connect(self.horizontalSlider_saturationFunction)
        self.spinBox_saturation.valueChanged.connect(self.spinBox_saturationFunction)
        self.horizontalSlider_sharpness.valueChanged.connect(self.horizontalSlider_sharpnessFunction)
        self.spinBox_sharpness.valueChanged.connect(self.spinBox_sharpnessFunction)
        #
        self.doubleSpinBox_conf.valueChanged.connect(self.doubleSpinBox_confFunction)
        self.doubleSpinBox_iounms.valueChanged.connect(self.doubleSpinBox_iounmsFunction)

        self.toolButton_openimage.clicked.connect(self.toolButton_openimageFunction)
        self.toolButton_openvideo.clicked.connect(self.toolButton_openvideoFunction)
        self.tabWidget_mode.tabBarClicked.connect(self.tabWidget_mode_clicked)
        self.pushButton_run.clicked.connect(self.pushButton_runFunction)
        # self.pushButton_inference.clicked.connect(self.pushButton_inferenceFunction)

        self.pushButton_video_stop.clicked.connect(self.pushButton_video_stopFunction)
        self.pushButton_video_run.clicked.connect(self.pushButton_video_runFunction)
        self.pushButton_infersize_apply.clicked.connect(self.pushButton_infersize_applyFunction)

        self.horizontalSlider_video.sliderReleased.connect(self.horizontalSlider_videosliderReleasedFunction)
        self.horizontalSlider_video.sliderPressed.connect(self.horizontalSlider_videosliderPressedFunction)
        self.horizontalSlider_video.sliderMoved.connect(self.horizontalSlider_videosliderMovedFunction)

        self.checkBox_view_img.stateChanged.connect(self.checkBox_view_imgFunction)
        self.checkBox_pcb_ocr.stateChanged.connect(self.checkBox_pcb_ocrFunction)

        self.comboBox_resoultion.activated[str].connect(self.comboBox_resoultionFunction)

        self.thread = VideoThread()
 
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.plate_info_signal.connect(self.get_plate_info)
        self.thread.video_info_signal.connect(self.video_info)
        self.thread.test_info_signal.connect(self.test_info)
        self.thread.right_data_signal.connect(self.right_data_info)
        #--

        self.run_infer = False


        #------------------------------data------------------------------#
        self.toolButton_openpt.clicked.connect(self.toolButton_openptFunction)
        self.pushButton_oneInference.clicked.connect(self.pushButton_oneInferenceFunction)


    def comboBox_resoultionFunction(self):
        if self.comboBox_resoultion.currentText() == '1280':
            self.thread.disply_width = 1280
            self.thread.display_height = 720
            self.canvas.disply_width = 1280
            self.canvas.display_height = 720
            self.canvas.resoultion()

        if self.comboBox_resoultion.currentText() == '960':
            self.thread.disply_width = 960
            self.thread.display_height = 720
            self.canvas.disply_width = 960
            self.canvas.display_height = 720
            self.canvas.resoultion()

        if self.comboBox_resoultion.currentText() == '640':
            self.thread.disply_width = 640
            self.thread.display_height = 480
            self.canvas.disply_width = 640
            self.canvas.display_height = 480
            self.canvas.resoultion()


    def pushButton_infersize_applyFunction(self):
        self.thread.yolo_PCB.set_infer_size(int(self.lineEdit_infersize.text()))
        self.thread.yolo_PNR.set_infer_size(int(self.lineEdit_infersize.text()))
        self.thread.yolo_PCBOCR.set_infer_size(int(self.lineEdit_infersize.text()))
        
        

    def pushButton_captureFunction(self):
        
        savefile=self.lineEdit_savefile.text() 
        savefilenum=self.lineEdit_savefilenum.text() 
        self.lineEdit_savefilenum.setText(str(int(savefilenum)+1))

        self.thread.capture(savefile,savefilenum)
        # br = [64, 128, 192]
        # ct = [128, 192, 255]
        # st = [64, 128, 192]
        # sp = [64, 128, 192]

        # savefile=self.lineEdit_savefile.text() 
        # savefilenum=self.lineEdit_savefilenum.text() 
        # self.lineEdit_savefilenum.setText(str(int(savefilenum)+1))

        

        # for bri in tqdm(br):
        #     self.thread.cap.set(cv2.CAP_PROP_BRIGHTNESS, bri/1)
        #     for cti in ct:
        #         self.thread.cap.set(cv2.CAP_PROP_CONTRAST, cti/1)
        #         for sti in st:
        #             self.thread.cap.set(cv2.CAP_PROP_SATURATION, sti/1)
        #             for spi in sp:
        #                 self.thread.cap.set(cv2.CAP_PROP_SHARPNESS, spi/1)
        #                 time.sleep(0.1)
        #                 save = './data/' + savefile + '_'+savefilenum + '_' + 'br' + str(bri) + 'ct' + str(cti) + 'st' + str(sti) + 'sp' + str(spi)
        #                 cv2.imwrite(save +'.jpg', self.thread.cv_img)


    @pyqtSlot(np.ndarray)
    def test_info(self,a):
        qt_img = self.convert_cv_qt(a, 320, 240)
        self.label_screen_pcbraw.setPixmap(qt_img)
        self.label_screen_pcbraw_ocr.setPixmap(qt_img)



    @pyqtSlot(tuple)
    def fiducial_signal(self, a):
        # self.lineEdit.setText(str(a))
        print(a)
        # self.thread.trainKP, self.thread.trainDesc, self.thread.imgMed, self.thread.im_aspect_ratio, self.thread.im_height, self.thread.im_width, self.thread.im_area,imCrop = search_feature(self.thread.cv_img,a,b)
        self.thread.trainKP, self.thread.trainDesc, self.thread.trainImg, self.thread.im_aspect_ratio, self.thread.im_height, self.thread.im_width, imCrop, self.thread.im_area = search_feature(self.thread.cv_img,a)

        fiducial_center, fiducial_box, angle, check = fiducial_marker(self.thread.cv_img, self.thread.trainKP, self.thread.trainDesc, self.thread.trainImg, self.thread.im_aspect_ratio, self.thread.im_height, self.thread.im_width,self.thread.im_area)

        if check:
            self.thread.fiducial_center_offset = fiducial_center
            self.thread.fiducial_angle_offset = angle
            print('good')
        else:
            print('again')
        



        self.update_image_fiducial(imCrop)
        # cv2.imshow('sdsd',imCrop)
    



















    @pyqtSlot(tuple)
    def roi_signal(self, a):
        print(a)
        self.thread.roi = a
        
        if self.thread.inference_mode == 1:
            self.thread.save_rightdata()
        # self.thread.trainKP, self.thread.trainDesc, self.thread.imgMed, self.thread.im_aspect_ratio, self.thread.im_height, self.thread.im_width, self.thread.im_area,imCrop = search_feature(self.thread.cv_img,a)
        # cv2.imshow('roi',imCrop)

        

    def lineEdit_infersizeFunction(self):
        self.thread.infersize=int(self.lineEdit_infersize.text())

    # data open pt
    def toolButton_openptFunction(self):
        path_ptfile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', filter='')
        print(path_ptfile[0])
        self.thread.yolo_data = yolov5(path_ptfile[0])
        self.lineEdit_ptfile.setText(path_ptfile[0])

    def pushButton_oneInferenceFunction(self):

        savefile=self.lineEdit_savefile.text() 
        savefilenum=self.lineEdit_savefilenum.text() 
        self.lineEdit_savefilenum.setText(str(int(savefilenum)+1))

        self.thread.savefile = './data/' + savefile + '_'+savefilenum
        self.thread.oneinference = True
        pass

    def pushButton_triggerFunction(self):
        # self.pushButton_run.setEnabled(True)
        if self.thread._run_flag == False:
            self.thread._run_flag = True
            self.pushButton_trigger.setText("■")
        else:
            self.thread._run_flag = False
            self.pushButton_trigger.setText("▶")


    def pushButton_fiducialFunction(self):
        self.canvas.canvas_mode = 'fiducial'
    def pushButton_fiducial_ocrFunction(self):
        self.canvas.canvas_mode = 'fiducial'

    def pushButton_ROIFunction(self):
        self.canvas.canvas_mode = 'ROI'
        self.thread.canvas_mode = 'ROI'
    def pushButton_ROI_ocrFunction(self):
        self.canvas.canvas_mode = 'ROI'
        self.thread.canvas_mode = 'ROI'

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def pushButton_camonFunction(self):
        print('Camera On')
        self.pushButton_camon.setEnabled(False)
        self.pushButton_camoff.setEnabled(True)
        self.thread.mode = 'cam'
        self.thread._run_flag = True
        self.thread.start()
        # time.sleep(5)
        # print('sdsd')
        # w = round(self.thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = round(self.thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = self.thread.cap.get(cv2.CAP_PROP_FPS)  # 카메라에 따라 값이 정상적, 비정상적
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # self.out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

    def pushButton_camoffFunction(self):
        # print('Stop')
        # self.record = False
        # self.out.release()
        print('Camera Off')
        self.pushButton_camoff.setEnabled(False)
        self.pushButton_camon.setEnabled(True)
        self.thread.stop()
        self.thread._run_flag = False
        self.thread.cap.release()

    def toolButton_openimageFunction(self):
        self.thread._run_flag = False
        self.thread.mode = 'image'
        path_imagefile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', filter='')
        print(path_imagefile[0])

        cv_img = cv2.imread(path_imagefile[0])
        qt_img = self.convert_cv_qt(cv_img, 960, 720)
        self.label_screen.setPixmap(qt_img)

    def toolButton_openvideoFunction(self):
        self.thread.mode = 'video'
        path_videofile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', filter='')
        print(path_videofile[0])
        print('video On')
        self.thread._run_flag = True
        self.thread.start()
        self.pushButton_video_run.setText("■")

        # self.thread.cap = cv2.VideoCapture('./samples/test.mp4')
        # self.thread.length = int(self.thread.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(self.thread.length)

    def pushButton_runFunction(self):
        # img=cv2.imread('./samples/test.jpg')

        self.pushButton_run.setEnabled(True)
        if self.run_infer == False:
            print('Run Inference')
            self.run_infer = True
            self.pushButton_run.setText("&STOP")
            self.thread.sw_inference = True
        else:
            self.run_infer = False
            print('Stop Inference')
            self.pushButton_run.setText("&RUN")
            self.thread.sw_inference = False

    # def pushButton_inferenceFunction(self):
    #     pass

    def pushButton_video_stopFunction(self):
        self.thread.video_ctr = 'stop'
        # self.thread.cap.release()

    def pushButton_video_runFunction(self):

        self.pushButton_video_run.setEnabled(True)
        if self.thread.video_ctr == 'stop':
            self.pushButton_video_run.setText("■")
            self.thread.video_ctr = 'play'
        else:
            self.pushButton_video_run.setText("▶")
            self.thread.video_ctr = 'stop'

    def horizontalSlider_videosliderMovedFunction(self):
        self.thread.video_ctr = 'stop'
        self.thread.framenum = self.horizontalSlider_video.value()

    def horizontalSlider_videosliderReleasedFunction(self):
        self.thread.video_ctr = self.video_pre_state

    def horizontalSlider_videosliderPressedFunction(self):
        self.video_pre_state = self.thread.video_ctr
        self.thread.video_ctr = 'stop'

    def checkBox_view_imgFunction(self):
        if self.checkBox_view_img.isChecked() == True:
            self.thread.view_img = True
        else:
            self.thread.view_img = False

    def checkBox_pcb_ocrFunction(self):
        if self.checkBox_view_img.isChecked() == True:
            self.thread.inference_mode = 10

    def tabWidget_mode_clicked(self, index):

        if index == 2:
            if self.checkBox_pcb_ocr.isChecked():
                self.thread.inference_mode = 10 #pcbocr
            else:
                self.thread.inference_mode = 2
        else:
            self.thread.inference_mode = index
        print(self.thread.inference_mode)

    '''
    #-- Get and print these values:
    #print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    #print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
    #print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
    #print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    #print("CAP_PROP_FOURCC  : '{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))))
    #print("CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    #print("CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)))
    #print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
    #print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
    #print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
    #print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
    #print("CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
    #print("CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)))
    #print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))
    #print("CAP_PROP_RECTIFICATION : '{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))
    #print("CAP_PROP_ISO_SPEED : '{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))
    #print("CAP_PROP_BUFFERSIZE : '{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))
    '''

    #-- CAP_PROP_FOCUS, 0~250, step=5, default=0

    def pushButton_autoipFunction(self):

        focus = int(self.thread.cap.get(cv2.CAP_PROP_FOCUS))
        brightness = int(self.thread.cap.get(cv2.CAP_PROP_BRIGHTNESS))
        contrast = int(self.thread.cap.get(cv2.CAP_PROP_CONTRAST))
        saturation = int(self.thread.cap.get(cv2.CAP_PROP_SATURATION))
        exposure = int(self.thread.cap.get(cv2.CAP_PROP_EXPOSURE))
        sharpness = int(self.thread.cap.get(cv2.CAP_PROP_SHARPNESS))

        print('focus : ' + str(focus))
        print('brightness : ' + str(brightness))
        print('contrast : ' + str(contrast))
        print('saturation : ' + str(saturation))
        print('exposure : ' + str(exposure))
        print('sharpness : ' + str(sharpness))

        self.thread.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.thread.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # AUTOFOCUS 켜기

        self.thread.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128/1)
        self.thread.cap.set(cv2.CAP_PROP_CONTRAST, 128/1)
        self.thread.cap.set(cv2.CAP_PROP_SATURATION, 128/1)
        self.thread.cap.set(cv2.CAP_PROP_SHARPNESS, 128/1)


    def horizontalSlider_focusFunction(self):
        self.spinBox_focus.setValue(self.horizontalSlider_focus.value())
        self.focus = self.horizontalSlider_focus.value() *5
        self.thread.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.thread.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        # self.thread.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # self.change_focus_value.emit(self.focus)

    def spinBox_focusFunction(self):
        # print("Focus set:" + str(self.focus))
        self.horizontalSlider_focus.setValue(self.spinBox_focus.value())
        # print(self.thread.cap.get(cv2.CAP_PROP_FOCUS))

    #-- CAP_PROP_BRIGHTNESS, 0~255, step=1, default=128
    def horizontalSlider_brightnessFunction(self):
        self.spinBox_brightness.setValue(self.horizontalSlider_brightness.value())
        self.brightness = self.horizontalSlider_brightness.value()
        self.thread.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness/1)

    def spinBox_brightnessFunction(self):
        #print("Bright set: " + str(self.spinBox_brightness.value()))
        self.horizontalSlider_brightness.setValue(
            self.spinBox_brightness.value())
        #print(self.thread.cap.get(cv2.CAP_PROP_BRIGHTNESS))

    #-- CAP_PROP_CONTRAST, 0~255, step=1
    def horizontalSlider_contrastFunction(self):
        self.spinBox_contrast.setValue(self.horizontalSlider_contrast.value())
        self.contrast = self.horizontalSlider_contrast.value()
        self.thread.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast/1)

    def spinBox_contrastFunction(self):
        self.horizontalSlider_contrast.setValue(self.spinBox_contrast.value())

    #-- CAP_PROP_SATURATION, 0~255, step=1
    def horizontalSlider_saturationFunction(self):
        self.spinBox_saturation.setValue(
            self.horizontalSlider_saturation.value())
        self.saturation = self.horizontalSlider_saturation.value()
        self.thread.cap.set(cv2.CAP_PROP_SATURATION, self.saturation/1)

    def spinBox_saturationFunction(self):
        self.horizontalSlider_saturation.setValue(
            self.spinBox_saturation.value())

    #-- CAP_PROP_EXPOSURE, 3~2047, step=1, default=250
    def horizontalSlider_exposureFunction(self):
        self.spinBox_exposure.setValue(self.horizontalSlider_exposure.value())
        self.exposure = self.horizontalSlider_exposure.value()
        self.thread.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.thread.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def spinBox_exposureFunction(self):
        self.horizontalSlider_exposure.setValue(self.spinBox_exposure.value())

    #-- CAP_PROP_SHARPNESS, 0~255, step=1, default=128
    def horizontalSlider_sharpnessFunction(self):
        self.spinBox_sharpness.setValue(
            self.horizontalSlider_sharpness.value())
        self.sharpness = self.horizontalSlider_sharpness.value()
        self.thread.cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness/1)

    def spinBox_sharpnessFunction(self):
        self.horizontalSlider_sharpness.setValue(
            self.spinBox_sharpness.value())

    def doubleSpinBox_confFunction(self):
        self.thread.conf = self.doubleSpinBox_conf.value()

    def doubleSpinBox_iounmsFunction(self):
        self.thread.iounms = self.doubleSpinBox_iounms.value()    

    # cam 스레드 실행

    @pyqtSlot(tuple) #np.ndarray
    def update_image(self, cv_img):

        self.canvas.update_image(cv_img[0])
        s_fps = "%0.0f FPS" % cv_img[1]
        self.label_fps.setText(s_fps)
        # print(cv_img[2])
        if self.thread.sw_inference == 1:
            inf_s_fps = "%0.0f ms(CORE)" % cv_img[2]
        else:
            inf_s_fps = "  ms(CORE)"

        self.label_inf_fps.setText(inf_s_fps)  
        """Updates the image_label with a new opencv image"""
        self.horizontalSlider_video.setValue(self.thread.framenum)

        # qt_img = self.convert_cv_qt(cv_img, self.screensize[0], self.screensize[1])
        # self.label_screen.setPixmap(qt_img)

    @pyqtSlot(list)
    def get_plate_info(self, plate_infos):
    
        if len(plate_infos) == 1:
            qt_img = self.convert_cv_qt(self.emptybox, 150, 50)
            self.label_screen_plate_2.setPixmap(qt_img)
            self.label_car_type_2.setText('')
            self.label_plate_num_2.setText('')
            self.label_screen_plate_3.setPixmap(qt_img)
            self.label_car_type_3.setText('')
            self.label_plate_num_3.setText('')
            self.label_screen_plate_4.setPixmap(qt_img)
            self.label_car_type_4.setText('')
            self.label_plate_num_4.setText('')
        if len(plate_infos) == 2:
            qt_img = self.convert_cv_qt(self.emptybox, 150, 50)
            self.label_screen_plate_3.setPixmap(qt_img)
            self.label_car_type_3.setText('')
            self.label_plate_num_3.setText('')
            self.label_screen_plate_4.setPixmap(qt_img)
            self.label_car_type_4.setText('')

        if len(plate_infos) == 4:
            qt_img = self.convert_cv_qt(self.emptybox, 150, 50)
            self.label_screen_plate_4.setPixmap(qt_img)
            self.label_car_type_4.setText('')

           
        for i, plate_info in enumerate(plate_infos):
            if i == 0:
                car_type, use_type, plate_num1, han, plate_num2, dst , points_list= plate_info
                self.label_car_type.setText(car_type)
                plate = plate_num1 + han + plate_num2
                self.label_plate_num.setText(plate)
                pp=str(points_list[0]) + str(points_list[1]) + str(points_list[2]) + str(points_list[3])
                self.listWidget.addItem((str(plate) + pp))
                self.listWidget.scrollToBottom()
                qt_img = self.convert_cv_qt(dst, 150, 50)
                self.label_screen_plate.setPixmap(qt_img)

            if i == 1:
                car_type, use_type, plate_num1, han, plate_num2, dst , points_list= plate_info
                self.label_car_type_2.setText(car_type)
                plate = plate_num1 + han + plate_num2
                self.label_plate_num_2.setText(plate)
                pp=str(points_list[0]) + str(points_list[1]) + str(points_list[2]) + str(points_list[3])
                self.listWidget.addItem((str(plate) + pp))
                self.listWidget.scrollToBottom()
                qt_img = self.convert_cv_qt(dst, 150, 50)
                self.label_screen_plate_2.setPixmap(qt_img)

            if i == 2:
                car_type, use_type, plate_num1, han, plate_num2, dst , points_list= plate_info
                self.label_car_type_3.setText(car_type)
                plate = plate_num1 + han + plate_num2
                self.label_plate_num_3.setText(plate)
                pp=str(points_list[0]) + str(points_list[1]) + str(points_list[2]) + str(points_list[3])
                self.listWidget.addItem((str(plate) + pp))
                self.listWidget.scrollToBottom()
                qt_img = self.convert_cv_qt(dst, 150, 50)
                self.label_screen_plate_3.setPixmap(qt_img)

            if i == 3:
                car_type, use_type, plate_num1, han, plate_num2, dst , points_list= plate_info
                self.label_car_type_4.setText(car_type)
                plate = plate_num1 + han + plate_num2
                self.label_plate_num_4.setText(plate)
                pp=str(points_list[0]) + str(points_list[1]) + str(points_list[2]) + str(points_list[3])
                self.listWidget.addItem((str(plate) + pp))
                self.listWidget.scrollToBottom()
                qt_img = self.convert_cv_qt(dst, 150, 50)
                self.label_screen_plate_4.setPixmap(qt_img)
        
        if len(self.listWidget) > 100:
            for i in range(len(self.listWidget)-100):
                self.listWidget.takeItem(0)


            # cv2.imshow('sdsd',dst)

    @pyqtSlot(tuple)
    def video_info(self, info):
        length, fps = info
        self.horizontalSlider_video.setRange(0, length)
        self.horizontalSlider_video.setSingleStep(1)

    # cvimage -> Qtimage
    def convert_cv_qt(self, cv_img, disply_width, display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_image_fiducial(self, img):
        qt_img = self.convert_cv_qt(img, 100, 100)
        self.label_screen_fiducial.setPixmap(qt_img)
        self.label_screen_fiducial_ocr.setPixmap(qt_img)

    @pyqtSlot(np.ndarray) #np.ndarray
    def right_data_info(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img, 190, 160)    
        self.label_screen_rightdata.setPixmap(qt_img)
        self.label_screen_rightdata_ocr.setPixmap(qt_img)

if __name__ == "__main__":
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    #WindowClass의 인스턴스 생성
    myWindow = WindowClass()
    #프로그램 화면을 보여주는 코드
    myWindow.show()
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
