## OpenCV 얼굴 인식 - 1

# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load img
img = cv2.imread('D:\\pythonProject1\\pofol\\red_boy.jpg')
plt.imshow(img)
plt.show() # --> opencv는 행렬이 BGR로 들어가 있음을 알 수 있다.

# convert RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

# convert to GrayScale
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray') # cmap을 지정해주지 않으면 노란색과 섞여서 나옴
plt.show()


## face detection
classifier = cv2.CascadeClassifier('.\\haarcascades\\haarcascade_frontalface_default.xml') # --> 앞면 인식할 수 있게 하는 데이터
# 얼굴을 찾아서 네모박스 생성
rects = classifier.detectMultiScale(gray,
                            scaleFactor=1.2, # 영상축소비율 (기본값 1.1)
                            minNeighbors=5 # 얼마나 많은 이웃 사각형이 검출되어야 최종 검출 영역으로 설정할지 지정 (기본값 3)
                            )
print('Faces Found: {}'.format(len(rects)))

# rectangle
print(rects) # --- [[140 146 244 244]]
print(rects[0]) # ---[140 146 244 244]
x, y, w, h = rects[0]

# draw a rectangle
cv2.rectangle(rgb, (x,y), (x+w,y+h), (0,255,0), 2) # (이미지이름, (x축+y축), (가로+세로), 사각형 색, 선 굵기)

# image plot
plt.imshow(rgb)
plt.show()



## OpenCV 얼굴 인식 - 2

# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load img
img = cv2.imread('D:\\pythonProject1\\pofol\\spurs_pic.jpg')
plt.imshow(img)
plt.show() # --> opencv는 행렬이 BGR로 들어가 있음을 알 수 있다.

# convert RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()



## face detection
classifier = cv2.CascadeClassifier('.\\haarcascades\\haarcascade_frontalface_default.xml') # --> 앞면 인식할 수 있게 하는 데이터
# 얼굴을 찾아서 네모박스 생성
rects = classifier.detectMultiScale(gray,
                            scaleFactor=1.2, # 영상축소비율 (기본값 1.1)
                            minNeighbors=5 # 얼마나 많은 이웃 사각형이 검출되어야 최종 검출 영역으로 설정할지 지정 (기본값 3)
                            )
print('Faces Found: {}'.format(len(rects)))

# rectangle
print(rects) # --- [[140 146 244 244]]
print(rects[0]) # ---[140 146 244 244]
# x, y, w, h = rects[0]
for (x,y,w,h) in rects:
    cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

# draw a rectangle

cv2.rectangle(rgb, (x,y), (x+w,y+h), (0,255,0), 2) # (이미지이름, (x축+y축), (가로+세로), 사각형 색, 선 굵기)

# image plot
plt.imshow(rgb)
plt.show()

# save pic
bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
cv2.imwrite('spurs_pic_face.jpg', bgr)
