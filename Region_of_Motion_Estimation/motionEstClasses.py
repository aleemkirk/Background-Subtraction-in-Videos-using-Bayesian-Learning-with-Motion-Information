#classes used to identify the regions of motion in a video frame
import cv2
import numpy as np
import math
import pandas

class imageSegmentor: #segments a frame into blocks of dimention x*y
    y, x = 0, 0
    originalImgShape = ()   #dimentions of original image. Used in image reconstruction
    paddedImgShape =  ()    #dimentions of padded image. Used in image reconstruction
    currentFrame = []
    currentFrameSegments, previousFrameSegments = [], []

    def __init__(self, firstFrame, x, y): #initalizes the dimentions of the blocks/segments
        self.x, self.y = x, y
        self.currentFrame = firstFrame
        self.originalImgShape = firstFrame.shape

    def changeFrame(self, image):
        self.currentFrame = image

    def padImg(self):
        #TODO if the image dimentions do not allow for an integer amount of blocks in either dimention pad the image until we get an integer number of blocks in all dimentions
        return


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
        for i in range(0, self.segmentor.currentFrameSegments, 1):
            if(np.sum(np.sum(self.segmentor.currentFrameSegments[i] - self.segmentor.previousFrameSegments[i], axis=0)) >= self.motionThresh): self.motionArr[i] = 1
            else: self.motionArr[i] = 0


class imageReconstructor:   #Reconstructs the image to show regions of motion

    def __init__(self):
        return

#--------------------Testing Section---------------------------
print("Everything is working!")