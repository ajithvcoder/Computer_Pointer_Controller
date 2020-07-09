'''
Head Pose Estimation
'''
from openvino.inference_engine import IENetwork as InferNetwork
from openvino.inference_engine import IECore as InferCore
import math

import cv2
import numpy as np


class FaceDetection:
    def __init__(self, model_name, device, threshold,extensions=None):
        self.faceLandMarkModelStructure = model_name + ".xml"
        self.faceLandMarkModelWeights = model_name + ".bin"

        try:
            self.model = InferNetwork(self.faceLandMarkModelStructure, self.faceLandMarkModelWeights)
        except Exception as e:
            raise ValueError("Please enter correct model path")

        self.dev = device
        self.thres = threshold
        self.extension = extensions
        self.cropFImage = None
        self.facePoints = None
        self.pImage = None
        self.network = None
        self.ffCoord = None
        self.res = None
        self.inName = next(iter(self.model.inputs))
        self.inShape = self.model.inputs[self.inName].shape
        self.outName = next(iter(self.model.outputs))
        self.outShape = self.model.outputs[self.outName].shape

    def load_model(self):
        self.model = InferNetwork(self.faceLandMarkModelStructure, self.faceLandMarkModelWeights)
        self.core = InferCore()
        supLayers = self.core.query_network(network=self.model, device_name=self.dev)
        unSupLayers = [k for k in self.model.layers.keys() if k not in supLayers]
        if len(unSupLayers) > 0:
            print("layers which are notsupported are found")
            print("Seeking Extension")
            self.core.add_extension(self.extension, self.dev)
            supLayers = self.core.query_network(network=self.model, device_name=self.dev)
            unSupLayers = [k for k in self.model.layers.keys() if k not in supLayers]
            if len(unSupLayers) > 0:
                print("Unsupported layers found even after adding extension")
                exit(1)
        self.network = self.core.load_network(network=self.model, device_name=self.dev, num_requests=1)

    def preprocess_input(self, image):
        img = cv2.resize(image, (self.inShape[3], self.inShape[2]))
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape)
        return img

    def preprocess_output(self, outputs, img):
        facePoints = []
        out = outputs[self.outName][0][0]
        for box in out:
            confidence = box[2]
            if confidence >=self.thres:
                xmin = int(box[3]* img.shape[1])
                ymin = int(box[4]*img.shape[0])
                xmax = int(box[5]* img.shape[1])
                ymax = int(box[6]*img.shape[0])
                facePoints.append([xmin,ymin,xmax,ymax])
        return facePoints


    def predict(self, img):
        self.pImage = self.preprocess_input(img)
        self.res = self.network.infer({self.inName:self.pImage})
        self.facePoints = self.preprocess_output(self.res,img)
        if len(self.facePoints) == 0:
            print("nexframe is being processed as no facepoints are detected")
            return 0,0
        self.ffCoord = self.facePoints[0]
        cropFImage = img[self.ffCoord[1]:self.ffCoord[3],self.ffCoord[0]:self.ffCoord[2]]
        return self.ffCoord,cropFImage






