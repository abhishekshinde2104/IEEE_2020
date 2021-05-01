import cv2
import numpy as np

def order_points(pts):
    pts = pts.reshape((4,2))
    rect = np.zeros((4,2),dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

image = cv2.imread("test_images/real1.jpg")
#we will resize the image as opencv doesnt perform well in bigger images
image = cv2.resize(image,(1300,800))

#original image stored and we will do operations on copied image
orig = image.copy()

#Gray scale image :
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Guasssian Blur
#technique to smoothen out your image
blurred = cv2.GaussianBlur(gray,(5,5),0)
#2nd arg is a kernel size which specifies a 2D matrix of a 5x5 matrix and this matrix is then moved on your image
#and a dot wise matrix multiplication would take place
#3rd arg is SIGMA which determines how much blur would take place

########
#returns,thresh=cv2.threshold(gray,80,255,cv2.THRESH_BINARY)

#contours,hierachy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
########


#canny edge detection
edged = cv2.Canny(blurred,30,50)
#30,50 are minimum and maximum values which are basically thresholding values

#Now we need to remove the noisee with find countours function
#so to extract the boundary
#returns the image,countours and the hierarchy

contours,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#1st arg is output of canny edged detection
#2nd arg is the retrival mtd
#3rd arg is the approximation model


#Next thing is to extract boundary which is the biggest countour
#so we sort in reverse order so wwe can find earlier
contours = sorted(contours,key=cv2.contourArea,reverse=True)

for c in contours:
    p = cv2.arcLength(c,True)
    #find sqaure within countours , True menans closed shapes
    #as this returned contour wouldnt be a perfect sqaure so we are going to approx is using the below function
    approx = cv2.approxPolyDP(c,0.02*p,True)

    if len(approx) == 4:
        target = approx
        break

approx = order_points(target)

pts = np.float32([[0,0],[800,0],[800,800],[0,800]])

op = cv2.getPerspectiveTransform(approx,pts)
dst = cv2.warpPerspective(orig,op,(800,800))



while True:
    cv2.imshow("Title",dst)

    #if we waited at least 1 milisecond and we pressed esc key then break and quit the process
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
