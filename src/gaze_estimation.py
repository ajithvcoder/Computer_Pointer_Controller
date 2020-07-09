'''
Head Pose Estimation
'''
from openvino.inference_engine import IENetwork as InferNetwork
from openvino.inference_engine import IECore as InferCore
import math

import cv2
import numpy as np

class GazeEstimation:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.gazeModelStructure = model_name + ".xml"
        self.gazeModelWeights = model_name + ".bin"
        try:
            self.model = InferNetwork(self.gazeModelStructure,self.gazeModelWeights)
        except Exception as e:
            raise ValueError("Please enter correct model path")

        self.dev = device
        self.extension = extensions
        self.inName = next(iter(self.model.inputs))

        self.outName = next(iter(self.model.outputs))
        self.outShape = self.model.outputs[self.outName].shape


    def load_model(self):
        self.model = InferNetwork(self.gazeModelStructure,self.gazeModelWeights)
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

    def preprocess_input(self, eyeLeft, eyeRight):
        eyeLeft = cv2.resize(eyeLeft,(60,60))
        eyeLeft = eyeLeft.transpose((2,0,1))
        eyeLeft = eyeLeft.reshape(1,*eyeLeft.shape)
        eyeRight = cv2.resize(eyeRight,(60,60))
        eyeRight = eyeRight.transpose((2,0,1))
        eyeRight = eyeRight.reshape(1,*eyeRight.shape)
        return eyeLeft,eyeRight


    def preprocess_output(self, outputs,hpout):
        val = hpout[2]
        out = outputs[self.outName][0]
        c = math.cos(val*math.pi/100)
        s = math.sin(val*math.pi/100)
        xVal = out[0]*c + out[1]*s
        yVal = out[1]*(c) - out[0]*s
        return (xVal,yVal),out

    def predict(self, eyeLeft,eyeRight,headPose):
        self.eyeLeft ,self.eyeRight = self.preprocess_input(eyeLeft,eyeRight)
        self.res = self.network.infer( inputs = {'left_eye_image':self.eyeLeft, 'right_eye_image':self.eyeRight, 'head_pose_angles':headPose})
        self.mpoints,self.gazeVector = self.preprocess_output(self.res,headPose)
        return self.mpoints,self.gazeVector





