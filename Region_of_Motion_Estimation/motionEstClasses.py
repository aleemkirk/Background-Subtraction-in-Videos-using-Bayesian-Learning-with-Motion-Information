#classes used to identify the regions of motion in a video frame
import numpy as np
import cv2 as cv2
import math
import pandas

class imageSegmentor: #segments a frame into blocks of dimention x*y
    segShape = ()             #tuple of segment/block dimentions
    originalImgShape = ()   #dimentions of original image. Used in image reconstruction
    paddedImgShape =  []    #dimentions of padded image. Used in image reconstruction
    currentFrame = []
    paddedImage = []
    currentFrameSegments, previousFrameSegments = [], []

    def __init__(self, firstFrame, x, y): #initalizes the dimentions of the blocks/segments
        self.segShape = (x, y)
        self.currentFrame = firstFrame
        self.originalImgShape = firstFrame.shape
        #determining the shape of the padded image
        if(self.currentFrame.shape[0]%self.segShape[0] == 0): rowDim = self.currentFrame.shape[0]
        else: rowDim = firstFrame.shape[0] + self.segShape[0] - (self.currentFrame.shape[0]%self.segShape[0])
        if(self.currentFrame.shape[1]%self.segShape[1] == 0): colDim = self.currentFrame.shape[1]
        else: colDim = firstFrame.shape[1] + self.segShape[1] - (self.currentFrame.shape[1]%self.segShape[1])
        self.paddedImgShape = tuple([rowDim, colDim])

    def changeFrame(self, image):
        self.currentFrame = image
        
    def padImg(self):
        self.paddedImage = np.zeros(self.paddedImgShape)
        self.paddedImage[:self.currentFrame.shape[0], :self.currentFrame.shape[1]] = self.currentFrame
        self.paddedImage = self.paddedImage.astype(self.currentFrame.dtype)
        return self.paddedImage

    def segmentImage(self):
        self.previousFrameSegments = self.currentFrameSegments #store previous image segments
        a = list()
        self.padImg()
        for y in range(0, int(self.paddedImgShape[0]/self.segShape[0]) , 1):
            for x in range(0, int(self.paddedImgShape[1]/self.segShape[1]), 1):
                seg = self.paddedImage[y*self.segShape[1]:y*self.segShape[1] + self.segShape[1], x*self.segShape[0]:x*self.segShape[0] + self.segShape[0]]
                a.append(seg)
        self.currentFrameSegments = np.asarray(a)
        return self.currentFrameSegments, self.previousFrameSegments

                


class motionDetector:   #Determins if motions has occured in a segment
    segmentor = None    #segmentor used to segment the video frames
    motionThresh = 0
    motionArr = []

    def __init__(self, segmentor, thresh):
        self.segmentor = segmentor
        self.motionThresh = thresh

    def updateSegmentor(self, segmentor):
        self.segmentor = segmentor
    
    def clearMotionArr(self):
        self.motionDetector = []

    def determineMotion(self):
        for i in range(0, self.segmentor.currentFrameSegments.shape[0], 1):
            if(np.sum(np.sum(self.segmentor.currentFrameSegments[i] - self.segmentor.previousFrameSegments[i], axis=0)) >= self.motionThresh): self.motionArr[i] = 1
            else: self.motionArr[i] = 0


class imageReconstructor:   #Reconstructs the image to show regions of motion

    def __init__(self):
        return

#--------------------Testing Section---------------------------
img = cv2.imread("img1.jpg", 0)
imgSegmentor = imageSegmentor(img, 100, 100)
print(imgSegmentor.originalImgShape)
print(imgSegmentor.paddedImgShape)
print((imgSegmentor.paddedImgShape[0]/imgSegmentor.segShape[0], imgSegmentor.paddedImgShape[1]/imgSegmentor.segShape[1]))
curr, prev = imgSegmentor.segmentImage()
print(imgSegmentor.paddedImage)
print(curr)
cv2.imshow("image", img)
cv2.imshow("padded image", imgSegmentor.paddedImage)
cv2.imshow("segment", curr[69])
cv2.waitKey(0)
cv2.destroyAllWindows()
