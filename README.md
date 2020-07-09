# Computer Pointer Controller

Computer Pointer Controller is a application which is used to control the movement of mouse pointer with the movement of eyes
and the position of head. In this application we give video as input and estimate the position of eyes in respect to it .


## Project Set Up and Installation

Install OpenVINO tool kit  - [Procedure](https://docs.openvinotoolkit.org/latest/)

Create a virtual environment with conda or venv


Initalize OpenVINO environment :
    For Windows :
    
        cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
        setupvars.bat
        
Download pretrained models with model downloader
    
    [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
    [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
    [Facial Landmark Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
    [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
   
For Windows 
 
Face Detection 
    
    python "C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/downloader.py" --name "face-detection-adas-binary-0001"
  
LandMark Regression 
    
    python "C:/Program Files (x86)/IntelSWTools/openvin/deployment_tools/tools/model_downloader/downloader.py" --name "landmarks-regression-retail-0009"

Head Pose Estimation 
    
    python "C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/downloader.py" --name "head-pose-estimation-adas-0001"

Gaze Estimation 
    
    python "C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/downloader.py" --name "gaze-estimation-adas-0002"

Clone repository 
    
    git clone 
    
Install the requirements 
    
    cd Computer_Pointer_Controller
    pip install -r requirements.txt
    
## Demo

Structural Command: 

    python <path to main.py> -fd <path to face detection directory> -fl <path to landmarks regression retail directory> -hp <path to head pose estimation directory> -ge <path to gaze estimation directory> -i <path to input video> -d CPU

Raw Command 

    python main.py -fd "../../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001" -fl "../../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -hp "../../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -ge "../../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" -i  ../bin/demo.mp4 -d CPU


## Documentation

-h --help : heap message 
-fd : (Mandatory) : Path to (.xml) of Face detection model 
-fl : (Mandatory) : Path to (.xml) of Facial Landmark model 
-hp : (Mandatory) : Path to (.xml) of Head pose Estimation model 
-ge : (Mandatory) : Path to (.xml) of Gaxe estimation model
-i  : (Mandatory) : Path to video file 
-flags : (Optional) : Specify flags fd,fl,hp,ge for vizuvalization of each model 
-d : (Optional) : device type

## Benchmarks

Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz

FP32 

Total Model Load Time : 856.46 ms
Total Inference Time : 45.5 seconds
FPS : 0.7257 frame/second



## Results

This model helps in moving the mouse pointer in accodance with the eye and head

## Stand Out Suggestions

Measured the time of each model as told in the suggestion space 

