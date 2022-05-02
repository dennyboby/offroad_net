import cv2
import numpy as np
import glob


def videoToimg(success,image):
    count = 0   
    while success:
        cv2.imwrite("images/frame%d.jpg" % count, image)     # save frame as JPEG file   
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        print(count)


if __name__ == '__main__':

    vidcap = cv2.VideoCapture('test.mkv')
    success,image = vidcap.read()
    videoToimg(success,image)



# img_array = []
# for filename in glob.glob('/home/fearless/RBE502/final_project/*.jpg'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)


# out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
# # cv2.VideoWriter_fourcc(*'DIVX')
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()