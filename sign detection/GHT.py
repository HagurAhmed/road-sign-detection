import  cv2
import  numpy as np

img = cv2.imread('test.png',1)
cv2.imshow("test",img)
#img = cv2.medianBlur(img,5)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv,np.array([0, 100, 100 ]) , np.array([10, 255, 255 ]))
mask2 = cv2.inRange(hsv, np.array([160, 100, 100 ]), np.array([179, 255, 255]))
mask3 = cv2.inRange(hsv,np.array([100,150,0]),np.array([140,255,255]))
mask = mask1+mask2 + mask3
#mask = cv2.max(mask1, mask2)
img = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((5, 5), np.float32) / 20
#img = cv2.filter2D(img, -1, kernel)
#img = cv2.medianBlur(img, 5)
cv2.imshow("h1",img)
cv2.waitKey(0)


