import os
import cv2
import bpy
import time
import struct
import itertools

from math import radians, ceil
from mathutils import Vector, Euler, Matrix

import numpy as np
from PIL import Image
from mediapipeHolisticWebcamMocapNET import MediaPipeHolistic
from MocapNET import easyMocapNETConstructor
from folderStream import resize_with_padding
from MocapNETVisualization import visualizeMocapNETEnsemble


scale = 0.01
needless = ('abdomen', 'chest', 'toe1-2.r', 'toe5-3.r', 'toe1-2.l', 'toe5-3.l', 'head', 'eye.l', 'eye.r')
needless = ()

needed = ('hip', 'hip', 'abdomen', 'chest', 'head', 'rThigh', 'rShin', 'rFoot', 'toe1-2.R', 'toe5-3.R', 
          'lThigh', 'lShin', 'lFoot', 'toe1-2.L', 'toe5-3.L', 'neck1', 'eye.l', 'eye.r', 'rShldr', 'rForeArm', 'rHand', 'lShldr', 'lForeArm', 'lHand')

# needed = ('hip', 'hip', 'head', 'rThigh', 'rShin', 'rFoot', 'lThigh', 'lShin', 'lFoot', 'rShldr',
#           'rForeArm', 'rHand', 'lShldr', 'lForeArm', 'lHand')

# Import blend file
bpy.ops.wm.open_mainfile(filepath="result3.blend")

# Render image from camera view conf
context = bpy.context
bpy.data.scenes["Scene"].render.image_settings.file_format = 'BMP'
context.scene.render.filepath = '/media/ramdisk/RenderResult'

FPS_average = 0
FPS_average_counter = 0.01


# BVH batching
def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    val_it = iter(iterable.values())
    key_it = iter(iterable.keys())
    while (batch := tuple(itertools.islice(val_it, n))) and (name := tuple(itertools.islice(key_it, n))[0].split('_', 1)[0]):
        if name in needless:
            continue
        yield batch


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
#########################


cap = cv2.VideoCapture(0)

ret, image = cap.read()

bones = bpy.data.armatures['out_from_github'].bones

count = 0
duration_f0 = 0
duration_f1 = 0
duration_f2 = 0
duration_f3 = 0

i = 0
prev_euler = Euler((0.0, 0.0, 0.0))

for obj in bpy.context.scene.objects:
    obj.select_set(False)

arm_data = bpy.data.armatures['out_from_github']
arm_ob = bpy.data.objects['out_from_github']

bpy.context.collection.objects.link(arm_ob)
arm_ob.select_set(True)
bpy.context.view_layer.objects.active = arm_ob
bpy.ops.object.mode_set(mode='EDIT', toggle=False)

bpy.context.view_layer.update()

bpy.ops.object.mode_set(mode='OBJECT', toggle=True)


# Main loop
while True:
    start_all = time.time()

    ret, image = cap.read()
    # image = np.asarray(Image.open('dataset/jpgs/1.jpg'))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    image = resize_with_padding(image, 1280, 720)

    # Perform image processing with MediaPipeHolistic
    mocapNETInput, annotated_image = mp_holistic.convertImageToMocapNETInput(image)

    # Perform 3D joint prediction with MocapNET
    mocapNET3DOutput = mnet.predict3DJoints(mocapNETInput)
    mocapNETBVHOutput = mnet.outputBVH

    _, plotImage = visualizeMocapNETEnsemble(mnet, annotated_image, bvhAnglesForPlotting=mocapNETBVHOutput, plotBVHChannels=True, economic=True)

    
    # cv2.imwrite('/media/ramdisk/blue.bmp', image)

    # start = time.time()
    # bpy_image = bpy.data.images.load('/media/ramdisk/blue.bmp')
    # bpy.context.scene.node_tree.nodes['Image'].image = bpy_image
    i = 0
    prev_euler = Euler((0.0, 0.0, 0.0))   # might need
    
    for key, val in zip(needed, batched(mocapNETBVHOutput, 3)):
        bone_rest_matrix = arm_data.bones[key].matrix_local.to_3x3()

        temp = bone_rest_matrix[1].copy()
        bone_rest_matrix[1] = bone_rest_matrix[2]
        bone_rest_matrix[2] = temp

        bone_rest_matrix[2] = -1 * bone_rest_matrix[2]

        bone_rest_matrix_inv = Matrix(bone_rest_matrix)
        bone_rest_matrix_inv.invert()

        bone_rest_matrix_inv.resize_4x4()
        bone_rest_matrix.resize_4x4()

        if i == 0:
            bone_translate_matrix = Matrix.Translation((Vector(val) - Vector((0, 0, +40))) * scale)
            arm_ob.pose.bones[key].location = (
                bone_rest_matrix_inv @ bone_translate_matrix).to_translation()
        else:
            if i == 1:
                val = tuple(radians(x) for x in reversed(val))  # Order from ZYX -> XYZ
                val = (radians( val[0]), radians(- val[1]), radians(- val[2]),) 
                euler = Euler(val, 'XYZ')
            else:
                # Shift val to the left so ZXY -> XYZ
                val = tuple(radians(x) for x in val[1:] + val[:1])
                euler = Euler(val, 'YXZ')
            data_path = arm_ob.pose.bones[key]
            bone_rotation_matrix = euler.to_matrix().to_4x4()
            bone_rotation_matrix = (
                bone_rest_matrix_inv @
                bone_rotation_matrix @
                bone_rest_matrix
            )
            data_path.rotation_euler = bone_rotation_matrix.to_euler(data_path.rotation_mode, prev_euler)
            prev_euler = data_path.rotation_euler
        i += 1

    start = time.time()
    bpy.ops.render.render(write_still=True)
    duration_f1 += time.time() - start
    
    # render = Image.open('/media/ramdisk/RenderResult.png')
    render = cv2.imread('/media/ramdisk/RenderResult.bmp')
    
    start = time.time()
    gray2 = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)

    # anything greater than 1 becomes 255 (white)
    _, mask = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)
    
    render = cv2.bitwise_and(render, render, mask=mask)
    image = cv2.bitwise_and(image, image, mask=mask_inv)
    
    result = cv2.add(render, image)
    
    # image = Image.fromarray(image).convert("RGBA")
    # image.paste(render, (0, 0), mask=render)
    # image = np.asarray(image)
    
    duration_f2 += time.time() - start

    cv2.imshow('Plot Image', plotImage)
    cv2.imshow('Webcam Stream', result)
    # cv2.imwrite('delete.jpg', result)

    count += 1
    # counter += 1

    duration_f3 += time.time() - start_all

    # if (time.time() - start_time) > x:
    #     FPS = counter / (time.time() - start_time)
    #     print("FPS: ", FPS)
    #     FPS_average_counter += 1
    #     FPS_average += FPS
    #     counter = 0
    #     start_time = time.time()

bpy.ops.wm.save_as_mainfile(filepath='result_test.blend')

# print(FPS_average / FPS_average_counter)
# print(duration_f0 / count)
print(duration_f1 / count)
print(duration_f2 / count)
print(duration_f3 / count)
print(bpy.data.scenes["Scene"].render.engine)

cap.release()
cv2.destroyAllWindows()
