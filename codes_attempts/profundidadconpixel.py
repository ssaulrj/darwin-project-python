## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2 as cv
import time
from matplotlib import pyplot as plt

def imfill(thresh):
    # Función para rellenar el interior de un contorno
    # Para más detalles de esta función, consultar:
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    im_floodfill = thresh.copy()
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    return thresh | im_floodfill_inv

def circularity(contours):
    new_contours = []
    for c in contours:
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        try:
            circ = (4 * np.pi * area) / (perimeter * perimeter)
        except ZeroDivisionError:
            circ = 0
        if circ > 0.5:
            new_contours.append(c)
    return new_contours

if __name__ == "__main__":
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    #estaba en 640x480
    config = rs.config()
    height, width = 240, 424
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30) #3' framerate

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 0.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    time.sleep(2.0)
    #count = 1954
    # Streaming loop
    try:
        while True:
            global_threshold = 95
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            #print('Depth', depth_image.shape)
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            #grey_color = 153
            grey_color = 1 #Color de fondo
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            #print('Clipping', clipping_distance)
            #print('Profundidad', depth_image_3d)

            # Render images
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
            #images = np.hstack((bg_removed, depth_colormap))
            cv.namedWindow('Align Example', cv.WINDOW_AUTOSIZE)

            cv.imwrite('frameremoved.png', bg_removed)

            img = cv.imread('frameremoved.png', 1)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, thresh = cv.threshold(gray, global_threshold, 254, cv.THRESH_BINARY)
            im_floodfill = imfill(thresh)
            contours, hierachy = cv.findContours(im_floodfill, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            mask1 = np.zeros((height, width), np.uint8)
            cv.drawContours(mask1, contours, -1, (255, 255, 255), 3)

            new_contours = circularity(contours)
            mask2 = np.zeros((height, width), np.uint8)
            cv.drawContours(mask2, new_contours, -1, (255, 255, 255), 3)
            mask2 = imfill(mask2)
            cv.imshow('Contornos', mask1)
            # print(len(new_contours))
            if new_contours is not None:
                for c in new_contours:
                    area = cv.contourArea(c)
                    if area > 1000:
                        #framessi = framessi + 1
                        # print(contours)
                        # Calcular el centro a partir de los momentos
                        momt = cv.moments(c)
                        # print('yey')
                        if momt["m00"] != 0:
                            cx = int(momt['m10'] / momt['m00'])
                            cy = int(momt['m01'] / momt['m00'])
                        else:
                            cx, cy = 0, 0 # set values as what you need in the situation
                        # Dibujar el centro
                        # cv.circle(mask2, (cx, cy), 3, (0, 0, 255), -1)
                        #depth_image[cx,cy]
                        print('cx {} cy {} cz {}', cx, cy, depth_image[cy,cx] )
                        cv.circle(mask2, (cx, cy), 3, (0, 0, 255), 5)
                        # Get data scale from the device and convert to meters
                        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                        depth = depth_image * depth_scale
                        dist, _, _, _ = cv.mean(depth)
                        print('Detected a ball {} meters away.',dist)
                        #images = np.hstack((bg_removed, mask2))
                        # Show images
                        #cv.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                        #cv.imshow('RealSense', images)
                        #cv.imshow('RealSense', bg_removed)
                        #cv.imshow('im_floodfill', im_floodfill)
                        cv.imshow("Frame", mask2)
            key = cv.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv.destroyAllWindows()
                break
    finally:
        pipeline.stop()