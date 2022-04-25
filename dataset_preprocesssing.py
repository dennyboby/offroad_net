import numpy as np
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from imutils import paths

def class_segmentor(img,R,G,B):
    # img[:,:,2]=0 #R
    # img[:,:,1]=0 #G
    # img[:,:,0]=0 #B
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
                a=img[row,col] 
                if (a[0]!=B or a[1]!=G or a[2]!=R) :
                        img[row,col][0]=0
                        img[row,col][1]=0
                        img[row,col][2]=0

    return img


if __name__ == '__main__':

    # imagePaths=list(paths.list_images("G:\MS Courses\Deep Learning\Group Project\my\\RUGD_annotations_combined"))

    
    for i in range(1,7437):

        img=cv.imread("/home/denny/dl_project/RUGD_annotations_combined/img ("+str(i)+").png")
        R,G,B=[64,64,64]
        img=class_segmentor(img,R,G,B)
        print('Processing image:',i)
        name='home/denny/dl_project/asphalt_64_64_64/img ('+str(i)+').png'
        cv.imwrite(name,img)
        # cv.imshow('img',img)  
        i+=1










    # imagePaths=list(paths.list_images("G:\MS Courses\Deep Learning\Group Project\my\\RUGD_annotations_combined"))

    # i=1
    # for image in imagePaths:

    #     img=cv.imread(image)
    #     R,G,B=[102,102,0]
    #     img=class_segmentor(img,R,G,B)
    #     name='G:\MS Courses\Deep Learning\Group Project\my\RUGD_annotations_combined_OffroadNet\img ('+str(i)+').png'
    #     cv.imwrite(name,img)
    #     # cv.imshow('img',img)  
    #     i+=1

