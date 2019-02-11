import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def hsv_colorname(color):
    # red = np.uint8([[[255,0,0 ]]])
    # hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

    hsv_color={'blue':[[100,150,0],[140,255,255]],'red':[[170,50,50],[180,255,255],[0,50,50],[10,255,255]],\
               'white':[[0,0,255],[0,0,255],[0,0,255]],'black':[0, 0, 0]}
    hsv = hsv_color[color]
    return (np.array(hsv[0]),np.array(hsv[1]))
    
def BGR2HSV(img,color_names):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    i=0
    for color in color_names:
            
        lower,upper =hsv_colorname(color)
        
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        
        if i != 0:
            final_img += res
        else: 
            final_img = res
        i +=1    
    
    median = cv2.medianBlur(final_img,5)
    # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    # cv2.imshow("output",np.hstack([median,img]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return median

#using hough cicrle
def cicles_detection(image,original_img):
    tmp_c = cv2.imread('t_c.png',0)
    tmp_r = cv2.imread('t_r.png',0)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cir_list=[]
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3,300)  #2,400
    
    if circles  is not None:
        #print("hagur")
        circles = np.round(circles[0,:]).astype("int")
        #print(len(circles)) 
        
        for (x,y,r) in circles:
            #print(r)
            if  19 <=r<=60:
                tmp_c = np.ones((200,200,3), np.uint8)
                tmp_r = tmp_c.copy()
                tmp_c= cv2.circle(tmp_c,(90,90), 85, (0,0,255), 4)
                tmp_r= cv2.rectangle(tmp_r,(20,20), (160,160), (0,0,255), 4)
                
                # _,r2=template_matching(cv2.cvtColor(original_img[y-r-2:y+r+2,x-r-2:x+r+2], cv2.COLOR_BGR2GRAY),cv2.cvtColor(tmp_c, cv2.COLOR_BGR2GRAY))
                # _,r3=template_matching(cv2.cvtColor(original_img[y-r-2:y+r+2,x-r-2:x+r+2], cv2.COLOR_BGR2GRAY),cv2.cvtColor(tmp_r, cv2.COLOR_BGR2GRAY))
                # print(r2,r3 ,'tmp')
                if True: #r2 > r3:
                    print('yes')
                    cir_list.append([x,y,r])
                    cv2.circle(output,(x,y),r,(0,255,0),4)
                    cv2.rectangle(output,(x-5,y-5),(x+5,y+5),(0,128,255),-1) 
                    cv2.circle(original_img,(x,y),r,(0,255,0),4)
                    cv2.rectangle(original_img,(x-5,y-5),(x+5,y+5),(0,128,255),-1)   
        # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        # cv2.imshow("output",np.hstack([image,output]))
        # cv2.waitKey(0)    
    return output,original_img,cir_list 

# using contoure detection 
def rectangle_detection(img,original_img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(gray,127,255,1)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 10, 250)
   
    rec_list=[]
    _,contours,_ = cv2.findContours(edged,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        area = cv2.contourArea(cnt)
        if area <=200 :
            continue 
        if len(approx)==3:
            print ("triangle area",area)
            cv2.drawContours(img,[cnt],-1,(0,128,255),4)
            cv2.drawContours(original_img,[cnt],-1,(0,128,255),4)
        elif len(approx)==4:
            print ("square area",area )
            M = cv2.moments(cnt)
            rec_list.append([M['m10']/M['m00'],M['m01']/M['m00']])
            cv2.drawContours(img,[cnt],-1,(0,128,255),4)
            cv2.drawContours(original_img,[cnt],-1,(0,128,255),4)
        # elif ((len(approx) > 8) & (area > 30) ):
        #     print("circle",area)
        #     cv2.drawContours(img,[cnt],-1,(0,128,255),4)
        #     cv2.drawContours(f,[cnt],-1,(0,128,255),4)
            
    
    # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    # cv2.imshow('output',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img,original_img,rec_list
         
def template_matching(img,template):
    # cv2.imshow("hagur tmp",img)
    # cv2.waitKey(0)
    if img is None:
        return img,0
    img2 = img.copy()
    #template = cv2.imread('t_c.png',0)
    template = cv2.resize(template,img.shape[0:2])
    #template=cv2.resize(template,(50, 50))  #triangle
    #template=cv2.resize(template,(50, 70))  #rectangle
    w, h = template.shape[::-1]
    #All the 6 methods for comparison in a list
    methods = [#'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED']#, 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
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
        # cv2.imshow('d',img)
        # cv2.waitKey(0)
    return img,max_val

def invlidate_circles(rec_l,cir_l,frame):
    for x1,y1 in rec_l:
        for x2,y2,_ in cir_l:
            dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
            print("dis",dist)
            if dist > 90 :
                cv2.circle(frame,(x,y),r,(0,255,0),4)
    return frame        
            

# img = cv2.imread('p1.jpg',1)
# res = BGR2HSV(img,['blue','red','white'])
# template_matching( cv2.cvtColor(res, cv2.COLOR_BGR2GRAY))
# res11= cicles_detection(res)
# rectangle_detection(img)

cap = cv2.VideoCapture("d.mp4")
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('tested.avi',fourcc, 20.0,(1280,720) ,1)  

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==True:
        
        
        #frame=template_matching( cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        img = BGR2HSV(frame,['blue','red','white'])
        img,_,cir_l = cicles_detection(img,frame)
        img,frame,rec_l = rectangle_detection(img,frame)
        if len(rec_l) !=0 and len(cir_l)!=0:
            frame=invlidate_circles(rec_l,cir_l,frame)
        else:
            for x,y,r in cir_l:
                cv2.circle(frame,(x,y),r,(0,255,0),4)
        out.write(frame)
        # frame= cv2.resize(frame,(350,350))
        # img= cv2.resize(img,(350,350))
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow("output",np.hstack([frame,img]))
        #cv2.waitKey(0)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
