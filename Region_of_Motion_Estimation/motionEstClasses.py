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

    def __init__(self, firstFrame, y, x): #initalizes the dimentions of the blocks/segments
        self.segShape = (y, x)
        self.currentFrame = firstFrame
        self.originalImgShape = firstFrame.shape
        self.currentFrameSegments = []
        #determining the shape of the padded image
        if(self.currentFrame.shape[0]%self.segShape[0] == 0): colDim = self.currentFrame.shape[0]
        else: colDim = firstFrame.shape[0] + self.segShape[0] - (self.currentFrame.shape[0]%self.segShape[0])
        if(self.currentFrame.shape[1]%self.segShape[1] == 0): rowDim = self.currentFrame.shape[1]
        else: rowDim = firstFrame.shape[1] + self.segShape[1] - (self.currentFrame.shape[1]%self.segShape[1])
        self.paddedImgShape = tuple([colDim, rowDim])

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
                seg = self.paddedImage[y*self.segShape[0]:y*self.segShape[0] + self.segShape[0], x*self.segShape[1]:x*self.segShape[1] + self.segShape[1]]
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
        self.motionArr = np.zeros(int(self.segmentor.paddedImgShape[0]/self.segmentor.segShape[0]) * int(self.segmentor.paddedImgShape[1]/self.segmentor.segShape[1]), dtype=int)

    def updateSegmentor(self, segmentor):
        self.segmentor = segmentor
    
    def clearMotionArr(self):
        self.motionDetector = np.zeros(int(self.segmentor.paddedImgShape[0]/self.segmentor.segShape[0]) * int(self.segmentor.paddedImgShape[1]/self.segmentor.segShape[1]), dtype=int)

    def determineMotion(self):
        self.clearMotionArr()
        for i in range(0, self.segmentor.currentFrameSegments.shape[0], 1):
            if(np.sum(np.sum(self.segmentor.currentFrameSegments[i] - self.segmentor.previousFrameSegments[i], axis=0)) >= self.motionThresh): self.motionArr[i] = 1
        return self.motionArr
    
    def setMotionArr(self, nArr):
        self.clearMotionArr()
        if(nArr.shape != self.motionArr.shape):
            return
        self.motionArr = nArr
        return self.motionArr
    
class imageReconstructor:   #Reconstructs the image to show regions of motion
    segmentor = None
    motionDetec = None
    outputImg = []


    def __init__(self, segmentor, motionDector):
        self.segmentor = segmentor
        self.motionDetec = motionDector
        self.outputImg = np.zeros(self.segmentor.paddedImgShape, dtype=self.segmentor.currentFrame.dtype)

    def reconstrucImg(self): 
        for i in range(0, self.motionDetec.motionArr.shape[0], 1):
            x = int(i%int(self.segmentor.paddedImgShape[1]/self.segmentor.segShape[1]))
            y = int(i/int(self.segmentor.paddedImgShape[1]/self.segmentor.segShape[1]))
            if(self.motionDetec.motionArr[i]):
                self.outputImg[y*self.segmentor.segShape[0]:y*self.segmentor.segShape[0] + self.segmentor.segShape[0], x*self.segmentor.segShape[1]:x*self.segmentor.segShape[1] + self.segmentor.segShape[1]] = np.full(self.segmentor.segShape, 255, dtype=self.segmentor.currentFrame.dtype)
        return self.outputImg

class motionRegionEst:
    inputImage = [] #input Image
    segShape  = [] #shape of the segments
    outputImg = [] #output image showing regions of motion
    motionThresh = 0
    segmentor = None
    motionDetec = None
    imgReconstructor = None

    def __init__(self, firstFrame, y, x, thresh):
        self.segmentor = imageSegmentor(firstFrame, y, x)
        self.segmentor.padImg()
        self.segmentor.segmentImage()
        self.motionDetec = motionDetector(self.segmentor, thresh)
        self.imgReconstructor = imageReconstructor(self.segmentor, self.motionDetec)
        self.motionThresh = thresh
        self.inputImage = firstFrame
        self.segShape = (y, x)

    def changeInputImg(self, image):
        self.inputImage = image
    
    def findMotion(self, img):
        self.segmentor.changeFrame(img)
        self.segmentor.padImg()
        self.segmentor.segmentImage()
        self.motionDetec.determineMotion()
        return self.imgReconstructor.reconstrucImg()





#--------------------Testing Section---------------------------
img = cv2.imread("img1.jpg", 0)
regionMotionDetec = motionRegionEst(img, 642, 200, 100)
img = cv2.imread("img1.jpg", 0)
cv2.imshow("Regions of motion", regionMotionDetec.findMotion(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
