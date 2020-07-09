'''
Head Pose Estimation
'''

import math

import cv2
import numpy as np
from openvino.inference_engine import IENetwork as InferNetwork
from openvino.inference_engine import IECore as InferCore

class FacialLandmark:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.faceLandMarkModelStructure = model_name + ".xml"
        self.faceLandMarkModelWeights = model_name + ".bin"
        try:
            self.model = InferNetwork(self.faceLandMarkModelStructure,self.faceLandMarkModelWeights)
        except Exception as e:
            raise ValueError("Please enter correct model path")

        self.dev = device
        self.extension = extensions
        self.inName = next(iter(self.model.inputs))
        self.inShape = self.model.inputs[self.inName].shape
        self.outName = next(iter(self.model.outputs))
        self.outShape = self.model.outputs[self.outName].shape


    def load_model(self):
        self.model = InferNetwork(self.faceLandMarkModelStructure,self.faceLandMarkModelWeights)
        self.core = InferCore()
        supLayers = self.core.query_network(network=self.model,device_name = self.dev)
        unSupLayers = [k for k in self.model.layers.keys() if k not in supLayers]
        if len(unSupLayers) > 0:
            print("layers which are notsupported are found")
            print("Seeking Extension")
            self.core.add_extension(self.extension,self.dev)
            supLayers = self.core.query_network(network=self.model,device_name=self.dev)
            unSupLayers = [k for k in self.model.layers.keys() if k not in supLayers]
            if len(unSupLayers)>0:
                print("Unsupported layers found even after adding extension")
                exit(1)
        self.network = self.core.load_network(network=self.model,device_name=self.dev,num_requests=1)

    def preprocess_input(self, image):
        img = cv2.resize(image, (self.inShape[3], self.inShape[2]))
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape)
        return img


    def preprocess_output(self, outputs,img):
        out = outputs[self.outName][0]
        lEyeXPoint = int(out[0]*img.shape[1])
        lEyeYPoint = int(out[1]*img.shape[0])
        rEyeXPoint = int(out[2]*img.shape[1])
        rEyeYPoint = int(out[3]*img.shape[0])

        return {'lEyeXPoint':lEyeXPoint,'lEyeYPoint':lEyeYPoint,
                'rEyeXPoint':rEyeXPoint,'rEyeYPoint':rEyeYPoint}


    def predict(self, img):
        self.img = self.preprocess_input(img)
        self.res = self.network.infer(inputs={self.inName:self.img})
        self.out = self.preprocess_output(self.res,img)
        eyeLeftXLow = self.out['lEyeXPoint']-10
        eyeLeftXHigh = self.out['lEyeXPoint']+10
        eyeLeftYLow = self.out['lEyeYPoint']-10
        eyeLeftYHigh = self.out['lEyeYPoint']+10

        eyeRightXLow = self.out['rEyeXPoint'] - 10
        eyeRightXHigh = self.out['rEyeXPoint'] + 10
        eyeRightYLow = self.out['rEyeYPoint'] - 10
        eyeRightYHigh = self.out['rEyeYPoint'] + 10

        self.ePoint = [[eyeLeftXLow,eyeLeftYLow,eyeLeftXHigh,eyeLeftYHigh],
                       [eyeRightXLow,eyeLeftYLow,eyeLeftXHigh,eyeLeftYHigh]]
        eyeLeftImg = img[eyeLeftXLow:eyeLeftXHigh,eyeLeftYLow:eyeLeftYHigh]
        eyeRightImg = img[eyeRightXLow:eyeRightXHigh,eyeRightYLow:eyeRightYHigh]
        return eyeLeftImg,eyeRightImg,self.ePoint








