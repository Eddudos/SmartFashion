import os
import cv2
import bpy
import time
import struct

from math import radians, ceil
from mathutils import Vector, Euler, Matrix

import numpy as np
from PIL import Image
from mediapipeHolisticWebcamMocapNET import MediaPipeHolistic
from MocapNET import easyMocapNETConstructor
from folderStream import resize_with_padding
from MocapNETVisualization import visualizeMocapNETEnsemble


# Import blend file
bpy.ops.wm.open_mainfile(filepath="result1.blend")

# Render image from camera view conf
context = bpy.context
context.scene.render.image_settings.file_format = 'BMP'
context.scene.render.filepath = '/media/ramdisk/RenderResult'


FPS_average = 0
FPS_average_counter = 0.01

#######################
mp_holistic = MediaPipeHolistic(doMediapipeVisualization=False)
mnet = easyMocapNETConstructor(
    engine="onnx",
    doProfiling=False,
    multiThreaded=True,
    doHCDPostProcessing=True,
    hcdLearningRate=0.001,
    hcdEpochs=99,
    hcdIterations=99,
    bvhScale=1.0,
    doBody=True,
    doFace=False,
    doREye=False,
    doMouth=False,
    doHands=False,
    addNoise=0.0
)

cap = cv2.VideoCapture(0)


ret, image = cap.read()

count = 0
duration_f1 = 0
duration_f2 = 0
duration_f3 = 0


bones = bpy.data.armatures['out_from_github'].bones

bvh_nodes = {}

for rest_bone in bpy.data.armatures['out_from_github'].bones:
    
    bone_rest_matrix = rest_bone.matrix_local.to_3x3()

    bone_rest_matrix_inv = Matrix(bone_rest_matrix)
    bone_rest_matrix_inv.invert()

    bone_rest_matrix_inv.resize_4x4()
    bone_rest_matrix.resize_4x4()
    
    bvh_nodes[rest_bone.name] = (bone_rest_matrix, bone_rest_matrix_inv)


arr = [i for i in range(72)]

while True:
    start = time.time()
    
    ret, image = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    image = resize_with_padding(image, 1280, 720)
    
    
    # Perform image processing with MediaPipeHolistic
    mocapNETInput, annotated_image = mp_holistic.convertImageToMocapNETInput(image)

    # Perform 3D joint prediction with MocapNET
    mocapNET3DOutput = mnet.predict3DJoints(mocapNETInput)
    mocapNETBVHOutput = mnet.outputBVH

    _, plotImage = visualizeMocapNETEnsemble(mnet, annotated_image, plotBVHChannels=False, economic=True)

    # start = time.time()
    cv2.imwrite('/media/ramdisk/blue.bmp', image)
    # duration_f1 += time.time() - start
    
    # start = time.time()
    bpy_image = bpy.data.images.load('/media/ramdisk/blue.bmp')
    bpy.context.scene.node_tree.nodes['Image'].image = bpy_image
    
    # duration_f2 += time.time() - start
    
    elem = 0
    
    # Надо бы по 3 элемента вытаскивать... можно так
    # arr = [  for i in mocapNETBVHOutput.values()]
    
    # Можно в tuple превращать для индексации... 
    # a = tuple(mocapNETBVHOutput.values())
    
    # Или с вот этим что-то придумать
    # for ind, out_value in enumerate(mocapNETBVHOutput.values()):
    #     arr[ind] = 
        
             
    # start = time.time()
    bpy.ops.render.render(write_still=True)
    image = cv2.imread('/media/ramdisk/RenderResult.bmp')
    bpy.data.images.remove(bpy_image)

    cv2.imshow('Webcam Stream', image)
    
    
    count += 1
    # counter += 1
    
    duration_f3 += time.time() - start

    # if (time.time() - start_time) > x:
    #     FPS = counter / (time.time() - start_time)
    #     print("FPS: ", FPS)
    #     FPS_average_counter += 1
    #     FPS_average += FPS
    #     counter = 0
    #     start_time = time.time()

# bpy.ops.wm.save_as_mainfile(filepath='result_test.blend')

# print(FPS_average / FPS_average_counter)
# print(duration_f1 / count)
# print(duration_f2 / count)
print(duration_f3 / count)

cap.release()
cv2.destroyAllWindows()
#######################




# 'hip_xposition':
# 114.46351774570394
# 'hip_yposition':
# -47.79586450396391
# 'hip_zposition':
# -223.34681186193848
# 'hip_zrotation':
# 15.009193555897355
# 'hip_yrotation':
# -12.926618920159683
# 'hip_xrotation':
# 5.742617545166354
# 'abdomen_zrotation':
# 0.0
# 'abdomen_xrotation':
# 0.0
# 'abdomen_yrotation':
# 0.0

