from typing import Any
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QPen, QPixmap, QPalette
from PyQt5.QtCore import QSize, Qt, QLine, QPoint, pyqtSignal, reset, pyqtSlot
from PyQt5.QtWidgets import*
import cv2
import numpy as np


class Screen(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    mousePressEvent_signal = pyqtSignal(object)
    mouseMoveEvent_signal = pyqtSignal(object)
    mouseReleaseEvent_signal = pyqtSignal(object)


    # 마우스 클릭
    def mousePressEvent(self, event):
        self.mousePressEvent_signal.emit(event)

    # 마우스 MOVE
    def mouseMoveEvent(self,event):
        self.mouseMoveEvent_signal.emit(event)
        
    # 마우스 RELEASE
    def mouseReleaseEvent(self,event):
        self.mouseReleaseEvent_signal.emit(event)


class Canvas(QtWidgets.QLabel):
    
    fiducial_signal = pyqtSignal(tuple)
    roi_signal = pyqtSignal(tuple)
   
    def setupUi(self):
        self.disply_width = 1280
        self.display_height = 720
        
        self.screen = Screen(self)
        self.screen.setBackgroundRole(QPalette.Base)
        # self.screen.setScaledContents(True)
        self.screen.setObjectName("screen")
        self.screen.resize(self.disply_width, self.display_height)

        self.scrollArea = QScrollArea(self)
        # self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.screen)
        # self.scrollArea.setVisible(False)
        self.scrollArea.resize(self.disply_width, self.display_height)
        self.scrollArea.setVisible(True)

        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def resoultion(self):
        self.screen.resize(self.disply_width, self.display_height)
        self.scrollArea.resize(self.disply_width, self.display_height)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.setAcceptDrops(True)

        self.scaleFactor = 1.0

        self.state_draw_rect = False

        # 마우스 이벤트 시작값
        self.past_x = None
        self.past_y = None
        # 마우스 이벤트 종료값
        self.present_x = None
        self.present_y = None

        self.py = 0
        self.px = 0

        # canvas mode
        self.canvas_mode = 'default'

        # create the video capture thread
        # connect its signal to the update_image slot
        # self.parent.change_pixmap_signal.connect(self.update_image)

        self.screen.mousePressEvent_signal.connect(self.s_mousePressEvent)
        self.screen.mouseMoveEvent_signal.connect(self.s_mouseMoveEvent)
        self.screen.mouseReleaseEvent_signal.connect(self.s_mouseReleaseEvent)

        self.gridx = [i*int(32*self.scaleFactor) for i in range(int((self.disply_width*self.scaleFactor)/32))]
        self.gridy = [i*int(32*self.scaleFactor) for i in range(int((self.display_height*self.scaleFactor)/32))]


    def zoomIn(self):
        self.scaleFactor *= 1.25
        self.scaleImage()
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), 1.25)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), 1.25)

    def zoomOut(self):
        self.scaleFactor *= 0.8
        self.scaleImage()
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), 0.8)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), 0.8)

    def scaleImage(self):
        self.gridx = [i*int(32*self.scaleFactor) for i in range(int((self.disply_width*self.scaleFactor)/32))]
        self.gridy = [i*int(32*self.scaleFactor) for i in range(int((self.display_height*self.scaleFactor)/32))]

        self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)
        self.screen.resize(self.rimg.size())
        self.screen.setPixmap(self.rimg)
        # self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        # self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        # print(type(cv_img))
        self.qt_img = self.convert_cv_qt(cv_img)

        
        self.rimg=self.qt_img.scaled(self.scaleFactor * self.qt_img.size(),Qt.KeepAspectRatio, Qt.FastTransformation)

        if self.canvas_mode == 'ROI':
            self.draw_grid(self.rimg)
            if self.state_draw_rect:
                self.draw_rect(self.rimg,int(self.past_x_grid*self.scaleFactor),int(self.past_y_grid*self.scaleFactor),int(self.x_grid*self.scaleFactor),int(self.y_grid*self.scaleFactor),Qt.green)

        if self.canvas_mode == 'fiducial':
            # if abs(self.x - self.px) > 0 or abs(self.y - self.py) > 0:
            # self.draw_line(self.rimg,int(self.x*self.scaleFactor),int(self.y*self.scaleFactor))
            if self.state_draw_rect:
                self.draw_rect(self.rimg,int(self.past_x*self.scaleFactor),int(self.past_y*self.scaleFactor),int(self.x*self.scaleFactor),int(self.y*self.scaleFactor),Qt.green)
                # self.py=self.y
                # self.px=self.x

        # self.screen.resize(self.rimg.size())
        self.screen.setPixmap(self.rimg)
    



    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio, Qt.FastTransformation)
        return QPixmap.fromImage(p)






    # 마우스 클릭
    @pyqtSlot(object)
    def s_mousePressEvent(self, event):
        self.past_x = int(event.x()/self.scaleFactor)
        self.past_y = int(event.y()/self.scaleFactor)

        tempx = []
        tempy = []
        
        for i in self.gridx:
            tempx.append(abs((self.x * self.scaleFactor) - i))
        self.past_x_grid = self.gridx[tempx.index(min(tempx))]

        for i in self.gridy:
            tempy.append(abs((self.y*self.scaleFactor) - i))
        self.past_y_grid = self.gridy[tempy.index(min(tempy))]


        self.state_draw_rect = True


    # 마우스 MOVE
    @pyqtSlot(object)
    def s_mouseMoveEvent(self,event):

        self.x =int(event.x()/self.scaleFactor)
        self.y =int(event.y()/self.scaleFactor)

        self.py=self.y
        self.px=self.x

        # if self.canvas_mode == 'fiducial':
        #     if abs(self.x - self.px) > 0 or abs(self.y - self.py) > 0:
                # self.draw_line(self.rimg,self.x*self.scaleFactor,int(self.y*self.scaleFactor))
                # if self.state_draw_rec == True:
                #     self.draw_rect(self.rimg,int(self.past_x*self.scaleFactor),int(self.past_y*self.scaleFactor),int(self.x*self.scaleFactor),int(self.y*self.scaleFactor),Qt.green)
                # self.py=self.y
                # self.px=self.x
        # if self.canvas_mode == 'ROI':
        #     if abs(self.x - self.px) > 0 or abs(self.y - self.py) > 0:
        #         # self.scaleImage()
        #         # self.draw_grid()
        #         # self.draw_line(self.x*self.scaleFactor,int(self.y*self.scaleFactor))
        #         if self.state_draw_rec == True:
        #             self.draw_rect(int(self.past_x*self.scaleFactor),int(self.past_y*self.scaleFactor),int(self.x*self.scaleFactor),int(self.y*self.scaleFactor),Qt.yellow)
        #         self.py=self.y
        #         self.px=self.x
        
        tempx = []
        tempy = []

        for i in self.gridx:
            tempx.append(abs((self.x * self.scaleFactor) - i))
        self.x_grid = self.gridx[tempx.index(min(tempx))]

        for i in self.gridy:
            tempy.append(abs((self.y*self.scaleFactor) - i))
        self.y_grid = self.gridy[tempy.index(min(tempy))]

            


    # 마우스 RELEASE
    @pyqtSlot(object)
    def s_mouseReleaseEvent(self,event):

        if self.canvas_mode == 'fiducial':
            self.canvas_mode = 'dafault'

            if self.state_draw_rect == True:
                self.state_draw_rect = False
                # self.draw_rect(int(self.past_x),int(self.past_y),int(event.x()),int(event.y()))

                # print(self.past_x,self.past_y)
                if int(self.past_x) > int(event.x()/self.scaleFactor):
                    x1 = int(event.x()/self.scaleFactor)
                    x2 = int(self.past_x)
                elif int(self.past_x) == int(event.x()/self.scaleFactor):
                    x1 = int(event.x()/self.scaleFactor)
                    x2 = int(event.x()/self.scaleFactor)+1
                else:
                    x1 = int(self.past_x)
                    x2 = int(event.x()/self.scaleFactor)

                if int(self.past_y) > int(event.y()/self.scaleFactor):
                    y2 = int(self.past_y)
                    y1 = int(event.y()/self.scaleFactor)
                elif int(self.past_y) == int(event.y()/self.scaleFactor):
                    y2 = int(self.past_y) + 1
                    y1 = int(event.y()/self.scaleFactor)
                else:
                    y1 = int(self.past_y)
                    y2 = int(event.y()/self.scaleFactor)


                # screen_ratio = self.disply_width/1280
                rect_point = [int(self.past_x/self.scaleFactor),int(self.past_y/self.scaleFactor),int(event.x()/self.scaleFactor),int(event.y()/self.scaleFactor)]
                

                
                self.rect_point= (x1,y1,x2-x1,y2-y1)
                self.fiducial_signal.emit(self.rect_point)
                
           
                # 마우스 이벤트 시작값 초기화
                self.state_draw_rect = False
                self.past_x = None
                self.past_y = None

        if self.canvas_mode == 'ROI':
            self.canvas_mode = 'dafault'
            same_error = False
            if self.state_draw_rect == True:
                self.state_draw_rect = False
                # self.draw_rect(int(self.past_x),int(self.past_y),int(event.x()),int(event.y()))

                # print(self.past_x,self.past_y)
                if int(self.past_x_grid) > int(self.x_grid/self.scaleFactor):
                    x1 = int(self.x_grid/self.scaleFactor)
                    x2 = int(self.past_x_grid)
                elif int(self.past_x_grid) == int(self.x_grid/self.scaleFactor):
                    same_error = True
                else:
                    x1 = int(self.past_x_grid)
                    x2 = int(self.x_grid/self.scaleFactor)

                if int(self.past_y_grid) > int(self.y_grid/self.scaleFactor):
                    y2 = int(self.past_y_grid)
                    y1 = int(self.y_grid/self.scaleFactor)
                elif int(self.past_y_grid) == int(self.y_grid/self.scaleFactor):
                    same_error = True
                else:
                    y1 = int(self.past_y_grid)
                    y2 = int(self.y_grid/self.scaleFactor)

                # rect_point = [int(self.past_x/self.scaleFactor),int(self.past_y/self.scaleFactor),int(event.x()/self.scaleFactor),int(event.y()/self.scaleFactor)]
                if same_error == False:
                    rect_point= (x1,y1,x2-x1,y2-y1)
                    self.roi_signal.emit(rect_point)

                # 마우스 이벤트 시작값 초기화
                self.state_draw_rect = False
                self.past_x = None
                self.past_y = None
                
                
    def draw_grid(self,img):

        tempx = []
        tempy = []
        
        painter = QPainter(img)
        painter.setPen(QPen(Qt.black, 1, Qt.DotLine))

        for i in self.gridx:
            painter.drawLine(i,0,i,int(self.display_height*self.scaleFactor))
            tempx.append(abs((self.x * self.scaleFactor) - i))

        painter.setPen(QPen(Qt.yellow, 1, Qt.DotLine))
        painter.drawLine(self.gridx[tempx.index(min(tempx))],0,self.gridx[tempx.index(min(tempx))],int(self.display_height*self.scaleFactor))

        painter.setPen(QPen(Qt.black, 1, Qt.DotLine))

        for i in self.gridy:
            painter.drawLine(0,i,int(self.disply_width*self.scaleFactor),i)
            tempy.append(abs((self.y*self.scaleFactor) - i))

        painter.setPen(QPen(Qt.yellow, 1, Qt.DotLine))
        painter.drawLine(0,self.gridy[tempy.index(min(tempy))],int(self.disply_width*self.scaleFactor),self.gridy[tempy.index(min(tempy))])

    
    def draw_rect(self,img,x1,y1,x2,y2, color):
        
        painter = QPainter(img)
        painter.setPen(QPen(color, 1, Qt.SolidLine))
        painter.drawRect(x1,y1,x2-x1,y2-y1)

        # tempx = []
        # tempy = []
        
        # painter = QPainter(img)
        # painter.setPen(QPen(Qt.black, 1, Qt.DotLine))

        # for i in self.gridx:
        #     painter.drawLine(i,0,i,int(480*self.scaleFactor))
        #     tempx.append(abs((self.x * self.scaleFactor) - i))

        # painter.setPen(QPen(Qt.yellow, 1, Qt.DotLine))
        # painter.drawLine(self.gridx[tempx.index(min(tempx))],0,self.gridx[tempx.index(min(tempx))],int(480*self.scaleFactor))


        # painter.setPen(QPen(Qt.black, 1, Qt.DotLine))

        # for i in self.gridy:
        #     painter.drawLine(0,i,int(640*self.scaleFactor),i)
        #     tempy.append(abs((self.y*self.scaleFactor) - i))

        # painter.setPen(QPen(Qt.yellow, 1, Qt.DotLine))
        # painter.drawLine(0,self.gridy[tempy.index(min(tempy))],int(640*self.scaleFactor),self.gridy[tempy.index(min(tempy))])
        # draw_rect(self,img,x1,y1,x2,y2, color):
        pass

    def draw_line(self,img, x, y):
        painter = QPainter(img)
        # painter.scale(self.scaleFactor,self.scaleFactor)

        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.setRenderHint(QPainter.HighQualityAntialiasing)
        # painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        # print(self.rimg.size().x())
        painter.drawLine(0,int(y),img.width(),int(y))
        painter.drawLine(int(x),0,int(x),img.height())


    # def draw_rect(self,img,x1,y1,x2,y2, color):
     

    #     painter = QPainter(img)
    #     # painter.scale(self.scaleFactor,self.scaleFactor)

    #     painter.setRenderHint(QPainter.Antialiasing)
    #     painter.setRenderHint(QPainter.HighQualityAntialiasing)
    #     painter.setRenderHint(QPainter.SmoothPixmapTransform)

    #     painter.setPen(QPen(color, 1, Qt.SolidLine))
    #     painter.drawRect(x1,y1,x2-x1,y2-y1)



# https://wiki.python.org/moin/PyQt/Painting%20an%20overlay%20on%20an%20image





# class CanvasLine(QtWidgets.QLabel):
    


#     def setupUi(self):
#         self.screen = Screen(self)
#         # self.screen.setBackgroundRole(QPalette.Base)
#         self.screen.setScaledContents(True)
#         self.screen.setObjectName("screen")
#         self.screen.resize(640, 480)



#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setupUi()
#         self.setAcceptDrops(True)

#     def paintEvent(self, e):
#         qp = QPainter()
#         qp.begin(self)
#         self.draw_line(qp)
#         qp.end()

#     def draw_line(self, qp):
#         qp.setPen(QPen(Qt.blue, 8))
#         qp.drawLine(30, 230, 200, 50)
#         qp.setPen(QPen(Qt.green, 12))
#         qp.drawLine(140, 60, 320, 280)
#         qp.setPen(QPen(Qt.red, 16))
#         qp.drawLine(330, 250, 40, 190)

