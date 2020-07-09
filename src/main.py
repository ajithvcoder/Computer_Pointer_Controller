import math
import time
import numpy as np
import os
import cv2

from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmark
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation

def bArgParser():
    parser = ArgumentParser()
    parser.add_argument("-fd","--face_detection_model",required=True,type=str,help="path for face detect model")
    parser.add_argument("-fl","--facial_landmarks_model",required=True,type=str,help="path for face landmark model")
    parser.add_argument("-hp","--head_pose_model",required=True,type=str,help="path for head pose model")
    parser.add_argument("-ge","--gaze_estimation_model",required=True,type=str,help="path for gaze estimation")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="specify device type cpu,gpu")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help="Prob threshold needed")
    parser.add_argument("-i","--input",required=True,type=str,help="path to video")
    parser.add_argument("-l","--cpu_extension",required=False,type=str,default=None,help="specify cpu extension")

    parser.add_argument("-flag","--visuvalization_flag",required=False,nargs="+",default=[],help="mention flags ")
    return parser

def alter(frame,mx1,my1,mz1,mz11,camMatrix,midx,midy):

    pointX2 = (mx1[0]/mx1[2])*camMatrix[0][0]+midx
    pointY2 = (mx1[1]/mx1[2])*camMatrix[1][1]+midy
    point2 = (int(pointX2),int(pointY2))
    cv2.line(frame,(midx,midy),point2,(0,0,255),2)

    pointX2 = (my1[0] / my1[2]) * camMatrix[0][0] + midx
    pointY2 = (my1[1] / my1[2]) * camMatrix[1][1] + midy
    point2 = (int(pointX2), int(pointY2))
    cv2.line(frame, (midx, midy), point2, (0, 0, 255), 2)

    pointX1 = (mz11[0] / mz11[2]) * camMatrix[0][0] + midx
    pointY2 = (mz11[1] / mz11[2]) * camMatrix[1][1] + midy
    point1 = (int(pointX1), int(pointY2))

    pointX2 = (mz1[0] / mz1[2]) * camMatrix[0][0] + midx
    pointY2 = (mz1[1] / mz1[2]) * camMatrix[1][1] + midy
    point2 = (int(pointX2), int(pointY2))
    cv2.line(frame,point1,point2,(255,0,0),2)
    cv2.circle(frame,point2,3,(255,0,0),2)

    return frame

#change var
def drawAxes(frame, midFace, vPointer,anchor, mdirection, scale, focalLength):
    vPointer *= np.pi/180.0
    anchor *= np.pi/180.0
    mdirection *= np.pi/180.0
    midx,midy = int(midFace[0]),int(midFace[1])
    xreach = np.array([[1,0,0],[0,math.cos(anchor),-math.sin(anchor)]])
    yreach = np.array([[math.cos(vPointer),0,-math.sin(vPointer)]])
    zreach = np.array([[math.cos(mdirection),-math.sin(mdirection),0],\
                       [math.sin(mdirection),math.cos(mdirection),0],[0,0,1]])
    reach = zreach @ yreach @ xreach
    camMatrix = buildCMatrix(midFace,focalLength)
    mx1 = np.array(([1*scale,0,0]),dtype='float32').reshape(3,1)
    my1 = np.array(([0,-1*scale,0]),dtype='float32').reshape(3,1)
    mz1 = np.array(([0,0,-1*scale]),dtype='float32').reshape(3,1)
    mz11 = np.array(([0,0,1*scale]),dtype='float32').reshape(3,1)

    temp = np.array(([0,0,0]),dtype='float32').reshape(3,1)
    temp[2] = camMatrix[0][0]
    mx1 = np.dot(reach,mx1)+temp
    my1 = np.dot(reach, my1) + temp
    mz1 = np.dot(reach, mz1) + temp
    mz11 = np.dot(reach, mz11) + temp

    frame = alter(frame,mx1,my1,mz1,mz11,camMatrix,midx,midy)
    return frame

def assign(focalLength,midx,midy,camMatrix):
    camMatrix[1][2],camMatrix[2][2],camMatrix[0][2],camMatrix[1][1],camMatrix[0][0] = midy, 1 , midx, focalLength,focalLength
    return camMatrix


def buildCMatrix(midFace,focalLength):
    midy = int(midFace[1])
    midx = int(midFace[0])
    camMatrix = np.zeros((3,3),dtype='float32')
    result = assign(focalLength,midx,midy,camMatrix)

    return result


def getPreferCount(modFaceDetection,modFaceialLandMarksDetection,modHeadPoseEstimation,modeGazeEstimation):
    st = time.time()
    modFaceDetection.load_model()
    print("Face Detect model loaded in {:.2f} ms".format((time.time()-st)*1000))
    p1 = time.time()
    modFaceialLandMarksDetection.load_model()
    print("Face LandMark Detection model loaded in {:.2f} ms".format((time.time() - p1) * 1000))
    p2 = time.time()
    modHeadPoseEstimation.load_model()
    print("Head Pose model loaded in {:.2f} ms".format((time.time() - p2) * 1000))
    p3 = time.time()
    modeGazeEstimation.load_model()
    print("Gaze Estimate model loaded in {:.2f} ms".format((time.time() - p3) * 1000))
    print("All Models are loaded")





if __name__ == '__main__':
    args = bArgParser().parse_args()
    inputFilePath = args.input

    vFlagsSet = args.visuvalization_flag
    if inputFilePath == "CAM":
        feedInput = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            print("ERROR : input path not valied")
            exit(1)
        feedInput = InputFeeder("video", inputFilePath)
    modelPaths = {'FD': args.face_detection_model,'FLD': args.facial_landmarks_model,'HPE': args.head_pose_model,
                  'GE': args.gaze_estimation_model}
    modFaceDetection = FaceDetection(model_name=modelPaths['FD'],
                                       device=args.device, threshold=args.prob_threshold,
                                       extensions=args.cpu_extension)
    modFaceialLandMarksDetection = FacialLandmark(model_name=modelPaths['FLD'],
                                               device=args.device, extensions=args.cpu_extension)
    modeGazeEstimation = GazeEstimation(model_name=modelPaths['GE'],
                                    device=args.device, extensions=args.cpu_extension)
    modHeadPoseEstimation = HeadPoseEstimation(model_name=modelPaths['HPE'],
                                            device=args.device, extensions=args.cpu_extension)
    mouseController = MouseController('medium', 'fast')
    st = time.time()
    getPreferCount(modFaceDetection,modFaceialLandMarksDetection,modHeadPoseEstimation,modeGazeEstimation)

    load_total_time= time.time() - st
    feedInput.load_data()
    print("input feeder loaded")

    count = 0
    stTimeInfer = time.time()
    print("Start inferencing on input video ")

    for subject, frame in feedInput.next_batch():
        if not subject:
            break
        pressed_key = cv2.waitKey(60)
        count += 1
        facePoints, faceImage = modFaceDetection.predict(frame.copy())
        if facePoints == 0:
            continue
        headPoseEstimationOutput = modHeadPoseEstimation.predict(faceImage)
        leftEyeImage, rightEyeImage, eyePoint = modFaceialLandMarksDetection.predict(faceImage)
        mousePoint, gazeVector = modeGazeEstimation.predict(leftEyeImage, rightEyeImage,
                                                        headPoseEstimationOutput)
        if len(vFlagsSet) != 0:
            prvwWindow = frame.copy()
            if 'fd' in vFlagsSet:
                if len(vFlagsSet) != 1:
                    prvwWindow = faceImage
                else:
                    cv2.rectangle(prvwWindow, (facePoints[0], facePoints[1]),
                                  (facePoints[2], facePoints[3]), (0, 150, 0), 3)
            if 'fl' in vFlagsSet:
                if not 'fd' in vFlagsSet:
                    prvwWindow = faceImage.copy()
                cv2.rectangle(prvwWindow, (eyePoint[0][0], eyePoint[0][1]),
                              (eyePoint[0][2], eyePoint[0][3]), (150, 0, 150))
                cv2.rectangle(prvwWindow, (eyePoint[1][0], eyePoint[1][1]),
                              (eyePoint[1][2], eyePoint[1][3]), (150, 0, 150))
            if 'hp' in vFlagsSet:
                cv2.putText(prvwWindow,
                            "vPointer:{:.1f} | anchor{:.1f} | mdirection:{:.1f}".format(headPoseEstimationOutput[0],
                                                                                        headPoseEstimationOutput[1],
                                                                                        headPoseEstimationOutput[2]),
                            (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
            if 'ge' in vFlagsSet:
                vPointer = headPoseEstimationOutput[0]
                anchor = headPoseEstimationOutput[1]
                mdirection = headPoseEstimationOutput[2]
                focalLength = 950.0
                scale = 50
                midFace = (faceImage.shape[1] / 2, faceImage.shape[0] / 2, 0)
                if 'fd' in vFlagsSet or 'fl' in vFlagsSet:
                    drawAxes(prvwWindow, midFace, vPointer, anchor, mdirection, scale, focalLength)
                else:
                    drawAxes(frame, midFace, vPointer, anchor, mdirection, scale, focalLength)
        if len(vFlagsSet) != 0:
            imgHor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(prvwWindow, (500, 500))))
        else:
            imgHor = cv2.resize(frame, (500, 500))
        cv2.imshow('Vizuvalization', imgHor)
        mouseController.move(mousePoint[0], mousePoint[1])

        if pressed_key == 27:
            print("Exit key pressed")
            break
    inferTime = round(time.time() - stTimeInfer, 1)
    fps = int(count) / inferTime
    print("counter {} seconds".format(count))
    print("total infer time {} seconds".format(inferTime))
    print("fps {} frame/second".format(fps))
    print("Video ended")

    feedInput.close()
    cv2.destroyAllWindows()
