'''
Head Pose Estimation
'''
from openvino.inference_engine import IENetwork as InferNetwork
from openvino.inference_engine import IECore as InferCore

import cv2
import numpy as np

class HeadPoseEstimation:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.headModelStructure = model_name + ".xml"
        self.headModelWeights = model_name + ".bin"
        try:
            self.model = InferNetwork(self.headModelStructure,self.headModelWeights)
        except Exception as e:
            raise ValueError("Please enter correct model path")

        self.dev = device
        self.extension = extensions
        self.inName = next(iter(self.model.inputs))
        self.inShape = self.model.inputs[self.inName].shape
        self.outName = next(iter(self.model.outputs))
        self.outShape = self.model.outputs[self.outName].shape


    def load_model(self):
        self.model = InferNetwork(self.headModelStructure,self.headModelWeights)
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
        img = cv2.resize(image,(self.inShape[3],self.inShape[2]))
        img = img.transpose((2,0,1))
        img = img.reshape(1,*img.shape)
        return img

    def preprocess_output(self, outputs):
        out = []
        out.append(outputs['angle_y_fc'].tolist()[0][0])
        out.append(outputs['angle_p_fc'].tolist()[0][0])
        out.append(outputs['angle_r_fc'].tolist()[0][0])
        return out


    def predict(self, image):
        self.image = self.preprocess_input(image)
        self.out = self.preprocess_output(self.network.infer(inputs={self.inName:self.image}))
        return self.out


