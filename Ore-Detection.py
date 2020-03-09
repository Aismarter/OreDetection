import cv2 as cv
import numpy as np

capture = cv.VideoCapture("矿石.mp4")
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)
# 角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
# KLT光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))
# 随机颜色
color = np.random.randint(0, 255, (100, 3))
# 读取第一帧
ret, old_frame = capture.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params, useHarrisDetector=False, k=0.04)
good_ini = p0.copy()


def caldist(a, b, c, d):
    return abs(a - c) + abs(b - d)


mask = np.zeros_like(old_frame)


def Canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t*2)
    return canny_output


def process(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # cv.imshow("hsv", hsv)
    line = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))

    mask = cv.inRange(hsv, (0, 43, 48), (45, 255, 255))
    cv.imshow("mask", mask)
    mask = cv.morphologyEx(mask, cv.MORPH_RECT, line)
    cv.imshow("mask", mask)
    binary = Canny_demo(mask)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        # x, y, w, h = cv.boundingRect(contours[c])
        # cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
        rect = cv.minAreaRect(contours[c])
        # cx, cy = rect[0]
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (255, 0, 0), 1)
        cv.drawContours(image, contours, c, (0, 255, 0), 1, 8)
    return image


def Convex_Hull(src):  # 凸包检测
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        ret = cv.isContourConvex(contours[c])
        points = cv.convexHull(contours[c])
        for i in range(len(points)):
            x1, y1 = points[i][0]
            cv.circle(src, (x1, y1), 3, (255, 0, 0), -1, 4, 0)
    return src


def LunKuo(src):  # 轮廓检测
    # dst = cv.edgePreservingFilter(src, sigma_s=100, sigma_r=0.8, flags=cv.RECURS_FILTER)
    # 对图像做滤波处理的
    # cv.imshow("dst", dst)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    binary = Canny_demo(gray)
    cv.imshow("gray_canny", binary)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        arclen = cv.arcLength(contours[c], True)  # 计算弧长， True表示闭合区域
        # if not (10 < area < 50) and (arclen > 2):
        #     continue
        cv.drawContours(src, contours, c, (0, 255, 0), 1, 8)
        rect = cv.minAreaRect(contours[c])
        cx, cy = rect[0]
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(src, [box], 0, (255, 0, 0), 1)
        cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (0, 0, 255), 1, 8, 0)

    return src


def canny(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    return canny_output


def line_Detection(src):
    binary = canny(src)
    # cv.imshow("line_binary", binary)

    linesP = cv.HoughLinesP(binary, 1, np.pi / 180, 150, None, 50, 10)
    # cv.HoughLinesP 霍夫变化的概论形式
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(src, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv.LINE_AA)
    return src


while True:
    ret1, frame1 = capture.read()
    image1 = frame1
    ret2, frame2 = capture.read()
    image2 = frame2
    ret3, frame3 = capture.read()
    image3 = frame3
    ret4, frame4 = capture.read()
    image4 = frame4
    if ret1 & ret2 & ret3 & ret4 is True:
        cv.imshow("video-input", frame1)
        result = process(image2)
        cv.imshow("result", result)
        cv.waitKey(50)
        detection = Convex_Hull(image1)  # 凸点检测
        cv.imshow("detection", detection)
        cv.waitKey(50)
        canny_detection = LunKuo(image3)
        cv.imshow("canny_detection", canny_detection)
        cv.waitKey(50)
        result = line_Detection(image4)
        cv.imshow("line_Detection", result)
        cv.waitKey(50)

        frame_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        # 计算光流
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 根据状态选择
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # 删除静止点
        k = 0
        for i, (new0, old0) in enumerate(zip(good_new, good_old)):
            a0, b0 = new0.ravel()
            c0, d0 = old0.ravel()
            dist = caldist(a0, b0, c0, d0)
            if dist > 2:
                good_new[k] = good_new[i]
                good_old[k] = good_old[i]
                good_ini[k] = good_ini[i]
                k = k + 1

        # 提取动态点
        good_ini = good_ini[:k]
        good_new = good_new[:k]
        good_old = good_old[:k]

        # 绘制跟踪线
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            image1 = cv.circle(image1, (a, b), 5, color[i].tolist(), -1)
        cv.imshow('KLT', cv.add(image1, mask))

        k = cv.waitKey(30) & 0xff
        if k == 27:
            cv.imwrite("flow.jpg", cv.add(image1, mask))
            break

        # 更新
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if good_ini.shape[0] < 40:
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            good_ini = p0.copy()

capture.release()
cv.destoryAllWindow()
