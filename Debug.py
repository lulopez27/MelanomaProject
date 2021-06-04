import cv2 #Image Library
import numpy as np #Numpy library
import matplotlib.pyplot as plt #matplotlib to show images
import statistics
import imutils
from statistics import mean

#https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python/35670916
#https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
#https://www.sciencedirect.com/science/article/abs/pii/S0895611116300398

def assymetry(imageName):
    image = cv2.imread(imageName) #read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Same image but in grayscale, this is done for calculating the threshold
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #get contours
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #fill in the contour
    cv2.drawContours(thresh, contours, -1, (255,255,255), thickness=cv2.FILLED)
    #find largest area contour and remove the smaller ones
    max_area = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area>max_area:
            max_cont = contours[i]
            max_area = area
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (area < max_area):
            cv2.drawContours(thresh, contours, -1, (0,0,0), 3)
    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate_bound(thresh, angle)
        print(findsymmetry(rotated))
    
#     image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #Get the threshold, separate the melanoma from healthy skin
    #It is done to binary, so all values are sent to either extreme, 0 or 1
#     x_first= 0 #get first white 
#     x_last = w-1
#     y_mid = int(np.ceil(h/2)) #assumption that image is well centred at y axis
    
    
#     #find the borders of the melanoma
#     while(thresh[y_mid][x_first] != 255):
#         x_first += 1
#     while(thresh[y_mid][x_last] != 255):
#         x_last -= 1
#     x_mid = int(np.ceil((x_last-x_first)/2 + x_first)) #the middle of the melanoma
#     #might have to redo this if I wish to truly get the middle of the object
    
#     #calculate how symmetric the object is:
#     tot_score = 0 #the lower the more symmetric it is
#     for i in range(0,h):
#         score_o = 0
#         for j in range(0,x_mid):
#             if( thresh[i][j] == 255):
#                 score_o += 1 
#         for j in range(x_mid,w):
#             if( thresh[i][j] == 255):
#                 score_o -= 1
#         tot_score += abs(score_o) #update the score
#     print(tot_score)
    
    
    thresh = thresh.clip(0, 255).astype('uint8') #Convert the threshold to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f, axarr = plt.subplots(1,2) #row,column
    axarr[0].imshow(image,cmap='hsv', vmin=0, vmax=255)
    axarr[1].imshow(thresh,cmap='gray', vmin=0, vmax=255) #show threshold
    
    
#     if(tot_score >= 1000):
#         print("This object is not simmetric")
#         return True #it is asymmetric
#     print("This object is simmetric")
#     return False #it is symmetric

def findsymmetry(im):
    #find the simetry lines along the x and y axis for the im image
    [y_max,x_max] = im.shape #get width and length of the image
    x = []
    xmina=[]
    xmaxa=[]
    y_max-=1
    x_max-=1
    for y in range(y_max):
        [x_next,xmin,xmax] = findmiddle(im[y,:],x_max)
        if(x_next != 0):
            x.append(x_next)
            xmina.append(xmin)
            xmaxa.append(xmax)
    x_line = int(mean(x))
    xassymetry = 0
    for x in range(len(xmina)):
        xassymetry += abs(abs(xmaxa[x]-x_line)-abs(xmina[x]-x_line))
    #centre line for y
    y = []
    ymina=[]
    ymaxa=[]
    for x in range(x_max):
        [y_next,ymin,ymax] = findmiddle(im[:,x],y_max)
        if(y_next != 0):
            y.append(y_next)
            ymina.append(ymin)
            ymaxa.append(ymax)
    y_line = int(mean(y))
    yassymetry = 0
    for x in range(len(xmina)):
        yassymetry += abs(abs(abs(ymaxa[x]-y_line)-abs(ymina[x]-y_line))
    return [yassymetry,xassymetry]
def findmiddle(line,max_):
    #return the halfway point between the first and last white pixel in a line, if there's no
    #white pixel it returns 0
    #line is the current line, can be row or column
    #max is the len of said line
    #min_ and max_ is the first and last white pixel
    min_=0
    while(line[min_] != 255):
        min_+=1
        if min_>=max_:#has no white pixels
            return [0,0,0]
    while(line[max_-1] != 255):
        max_-=1
    return [((max_-min_)/2)+min_,min_,max_]
imname = "Images/FirstImages/ISIC_0024792.png" #Name of the image to analyze, it uses relative pathing for image path
assymetry(imname)