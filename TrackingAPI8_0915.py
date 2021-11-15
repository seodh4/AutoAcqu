#
#--- Test Tracking --
#
import math
import numpy as np
import time
import cv2

detector = cv2.SIFT_create()

def search_feature(frame, roi):

    imCrop = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    im_height = (int(roi[1] + roi[3]) - int(roi[1])) / 2
    im_width = (int(roi[0] + roi[2]) - int(roi[0])) / 2
    im_aspect_ratio = im_width / im_height
    im_area = im_height * im_width


    sub_imCrop = cv2.resize(imCrop, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    trainImg = cv2.cvtColor(sub_imCrop, cv2.COLOR_BGR2GRAY)

    trainKP, trainDesc = detector.detectAndCompute(trainImg, None)

    return trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, imCrop, im_area


def fiducial_marker(frame2, trainKP, trainDesc, trainImg, im_aspect_ratio, im_height, im_width, im_area):

    box_1 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
    pre_H = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], np.int32)
    pTime = 0

    sub_frame2 = cv2.resize(frame2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    QueryImg = cv2.cvtColor(sub_frame2, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(queryDesc, trainDesc, k=2)

    cTime = time.time()
    sec = cTime - pTime
    pTime = cTime
    fps = 1 / (sec)
    s_fps = "FPS:%0.1f" % fps
    angle = 0
    center_x, center_y, det_img_aspect_ratio = 0, 0, 0

    # cv2.putText(frame2, s_fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1, cv2.LINE_AA)

    goodMatch = []
    for m, n in matches:
        if (m.distance < 0.70 * n.distance):
            goodMatch.append(m)

    MIN_MATCH_COUNT = len(goodMatch) - 3

    if MIN_MATCH_COUNT < 0:
        MIN_MATCH_COUNT = 0
    if (len(goodMatch) > MIN_MATCH_COUNT):
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))

        if len(tp) < 5 or len(qp) < 5:
            return None, None, 1, False
        else:
            pre_qp = qp
            pre_tp = tp
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)

        if H is None:
            H = pre_H
        else:
            pre_H = H

        h, w = trainImg.shape
        h_, w_ = QueryImg.shape
        trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)

        cod_query = queryBorder[0] * 2
        center_x_top = (int(cod_query[0][0]) + int(cod_query[3][0])) // 2
        center_y_top = (int(cod_query[0][1]) + int(cod_query[3][1])) // 2

        center_x_bot = (int(cod_query[1][0]) + int(cod_query[2][0])) // 2
        center_y_bot = (int(cod_query[1][1]) + int(cod_query[2][1])) // 2

        center_x = (center_x_top + center_x_bot) // 2
        center_y = (center_y_top + center_y_bot) // 2

        # cv2.line(frame2, (center_x, center_y), (center_x, center_y), (255, 255, 255), 3)

        point1 = np.array([cod_query[0], cod_query[1], cod_query[2], cod_query[3]], np.int32)

        b_x = (int(cod_query[3][0]) + int(cod_query[0][0])) // 2
        b_y = (int(cod_query[3][1]) + int(cod_query[0][1])) // 2

        # cv2.line(frame2, (center_x, center_y), (b_x, b_y), (0, 255, 255), 3)
        # cv2.line(frame2, (center_x, 0), (center_x, center_y), (255, 0, 255), 3)

        try:
            o1 = math.atan((0 - center_y) / (center_x - center_x))
        except ZeroDivisionError:
            o1 = 0
        
        try:
            o2 = math.atan((b_y - center_y) / (b_x - center_x))
        except ZeroDivisionError:
            o2 = 0
        
        if b_y <= center_y and b_x > center_x:
            angle = round(abs((o1 - o2) * 180 / math.pi), 2)
            angle = abs(angle - 90)
        elif b_y >= center_y and b_x > center_x:
            angle = round(abs((o1 - o2) * 180 / math.pi), 2)
            angle = angle + 90
        elif b_y >= center_y and b_x < center_x:
            angle = round(abs((o1 - o2) * 180 / math.pi), 2)
            angle = abs(angle - 270)
        elif b_y < center_y and b_x < center_x:
            angle = round(abs((o1 - o2) * 180 / math.pi), 2)
            angle = (angle + 270)
        elif b_x == center_x and b_y > center_y:
            angle = 180
        elif b_x == center_x and b_y < center_y:
            angle = 0

        rect = cv2.minAreaRect(point1)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        det_img_width = math.sqrt(math.pow((box[0][0] - box[3][0]), 2) + math.pow((box[0][1] - box[3][1]), 2))
        det_img_height = math.sqrt(math.pow((box[0][0] - box[1][0]), 2) + math.pow((box[0][1] - box[1][1]), 2))

        det_img_area = det_img_width * det_img_height

        if (im_width > im_height and det_img_width < det_img_height):
            temp = det_img_width
            det_img_width = det_img_height
            det_img_height = temp
        elif (im_width < im_height and det_img_width > det_img_height):
            temp = det_img_width
            det_img_width = det_img_height
            det_img_height = temp

        try:
            det_img_aspect_ratio = det_img_width / det_img_height
        except ZeroDivisionError:
            det_img_aspect_ratio = 0


        if ((det_img_aspect_ratio >= im_aspect_ratio * 0.9 and det_img_aspect_ratio <= im_aspect_ratio * 1.1) and (det_img_area >= (im_area * 4) * 0.9 and det_img_area <= (im_area * 4) * 1.3)) and ((center_x >= 0 and center_x <= 1280) and (center_y >= 0 and center_y <= 720)):
            box_1 = box
            return (center_x, center_y), [box_1],  angle, True
        # elif ((det_img_aspect_ratio < im_aspect_ratio * 0.9 and det_img_aspect_ratio > im_aspect_ratio * 1.1) and (det_img_area < (im_area * 4) * 0.9 and det_img_area > (im_area * 4) * 1.3))or (center_x > 1280 or center_x < 0) or (center_y < 0 or center_y > 720):
        else:
            return (center_x, center_y), [box_1],  angle, True
        # else:
        #     return  None, None, 2, False
    else:
        return None, None, 3, False


