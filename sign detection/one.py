import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
# Load an color image in grayscale
img = cv2.imread('messi5.jpg',1)
plt.imshow(img)
plt.show()
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.destroyWindow("image") 
#cv2.imwrite('messigray.png',img)
'''

'''
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''

'''
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480),1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
'''

'''
# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
plt.imshow(img)
plt.show()
'''

# image = cv2.imread('p1.jpg',1)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(3,3),0)
# edges = cv2.Canny(blur,100,200)
# cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)


# image = cv2.imread('p5.jpg',1)
# # print(image.shape)
# output = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # detect circles in the image
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2,400)  #3,300
    
# if circles  is not None:
#     circles = np.round(circles[0,:]).astype("int")
#     print(len(circles)) 
        
#     for (x,y,r) in circles:
#         #print(r)
#         if True:
#             cv2.circle(output,(x,y),r,(0,255,0),4)
#             cv2.rectangle(output,(x-5,y-5),(x+5,y+5),(0,128,255),-1)    
#     cv2.namedWindow('output', cv2.WINDOW_NORMAL)
#     cv2.imshow("output",np.hstack([image,output]))
#     cv2.waitKey(0)




from skimage import exposure

image = cv2.imread('p1.jpg',1)

#ratio = image.shape[0] / 300.0
orig = image.copy()
#image = imutils.resize(image, height = 300)
 
# convert the image to grayscale, blur it, and find edges
# in the image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# edged = cv2.Canny(gray, 30, 200)

# (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

# rects = [cv2.boundingRect(cnt) for cnt in cnts]
# rects = sorted(rects,key=lambda  x:x[1],reverse=True)


# for rect in rects:
#     x,y,w,h = rect
#     area = w * h
#     cv2.rectangle(orig,(x-5,y-5),(x+5,y+5),(0,128,255),-1)

# cv2.namedWindow('output', cv2.WINDOW_NORMAL)
# cv2.imshow("output",orig)
# cv2.wairectangle_detectiotKey(0)

# img = cv2.imread('p1.jpg',1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(gray,127,255,1)
# _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#     area = cv2.contourArea(cnt)
#     if area <=200 :
#         continue 
#     if len(approx)==3:
#         print ("triangle area",area)
#         cv2.drawContours(img,[cnt],0,(0,0,0),0)
#     elif len(approx)==4:
#         print ("square area",area )
#         cv2.drawContours(img,[cnt],0,(0,0,0),0)
    
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



img = cv2.imread('p1.jpg',0)
img2 = img.copy()
template = cv2.imread('p.png',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    cv2.imshow('d',img)
    cv2.waitKey(0)
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()









