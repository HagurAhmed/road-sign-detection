import cv2
from skimage import data
import numpy as np

def define_rect(image):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = [] # Starting and ending points
    win_name = "image" # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts

def modefie(points,frame,tmp,matcher,l_frames):
    if len(points) is 0:
        return frame
    output_frames=[]    
    length = len(l_frames)

    start = points[0]
    end = points[1]
    
    w,h=end[0]-start[0],end[1]-start[1]

    for i in range(length):
        frame = l_frames[i]
        i *=2
        w,h = w+i,h+i
        tmp = cv2.resize(tmp,(w, h))
        matcher =cv2.resize(matcher,(w, h))
        # print(tmp.shape,matcher.shape,frame.shape,w,h)
        # print(type(matcher[0,0]))   
        for i in range(0,w):
            for j in range(0,h):
                if (matcher[j,i] !=  (1,1,1)).all() or (tmp[j,i] == (0,0,255)).all():
                    frame[j+start[1],i+start[0]]=tmp[j,i]
        output_frames.append(frame)
    #m = frame[start[1]:end[1],start[0]:end[0]]
    #cv2.rectangle(frame,start, end, (255,255,240), 2) 
    # cv2.imshow("1",tmp)
    # cv2.imshow("2",matcher)
    # cv2.imshow("3",frame)
    # cv2.imshow("4",m)
    # cv2.waitKey(0)
    return output_frames



# # Prepare an image for testing
# lena = cv2.imread('p1.jpg',1) #data.lena() # A image array with RGB color channels
# #lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB) # Convert RGB to BGR

# # Points of the target window
# points = define_rect(lena)

# print("--- target window ---")
# print("Starting point is ", points[0])
# print("Ending   point is ", points[1])


img = np.ones((512,512,3), np.uint8)
img= cv2.circle(img,(250,250), 250, (0,0,255), -1)
# cv2.imshow('ff',img)
# cv2.waitKey(0)


cap = cv2.VideoCapture("road.avi") 
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('t2.avi',fourcc, 20.0,(976,740) ,1)   #(640,480)
l_frames=[]
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==True:
        cv2.imshow("frame",frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("a"): # Hit 'r' to replot the image
            points = define_rect(frame)
            l_frames.append(frame) 
        elif key == ord("s"):
            l_frames.append(frame)    
        elif key == ord('d'):
            frames=modefie(points,l_frames[0],cv2.imread('h1.png',1),img,l_frames)
            for frame in l_frames:
                out.write(frame)
                cv2.imshow('frame',frame)

        #cv2.imshow('frame',frame)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


#......................................................................
# img = np.ones((512,512,3), np.uint8)
# img= cv2.circle(img,(250,250), 250, (0,0,255), -1)
# cv2.imshow('ff',img)
# cv2.waitKey(0)

# # Prepare an image for testing
# lena = cv2.imread('p1.jpg',1) #data.lena() # A image array with RGB color channels
# #lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB) # Convert RGB to BGR

# # Points of the target window
# points = define_rect(lena)

# print("--- target window ---")
# print("Starting point is ", points[0])
# print("Ending   point is ", points[1])

# modefie(points,lena,cv2.imread('h1.png',1),img)