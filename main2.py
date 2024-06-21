import cv2
import numpy as np
image = cv2.imread('blobs.jpeg', 0)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100

params.filterByCircularity = True
params.minCircularity = 0.8

params.filterByConvexity = True
params.minConvexity = 0.9

params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)
number_of_blobs = len(keypoints)
print("Number of blobs :",number_of_blobs)

blank = np.zeros((1,1))
#drawKeypoints(input_image, key_points, output_image, colour, flag)
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

text="Number of Circular blobs : "+str(number_of_blobs)

#cv2.putText(image, text, position, font, fontScale, color, thickness)
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
cv2.imshow("Result", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()