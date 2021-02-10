# Shahriyar Mammadli
# Import required libraries
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

# The function obtains the landmarks of an eye and calculates the...
# ...eye aspect ratio of it.
def EAR(eye):
    # Calculate euclidean distance between upper and lower landmark...
    # ...couples.
    p2p6 = dist.euclidean(eye[1], eye[5])
    p3p5 = dist.euclidean(eye[2], eye[4])
    # Calculate euclidean distance between side landmarks.
    p1p4 = dist.euclidean(eye[0], eye[3])
    # Calculate ear aspect ratio
    ear = (p2p6 + p3p5) / (2.0 * p1p4)
    # return the eye aspect ratio
    return ear

# The function draws necessary details on the frame.
def drawOnStream(frame, rect, index, shape, leftEye, rightEye):
    # Draw a rectangle around the face.
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Write the face number next to the face.
    cv2.putText(frame, "Face #{}".format(index + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Draw the all face landmarks
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Draw the eyes using the corresponding landmarks.
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

# This function draws output of the process on the stream.
def drawResults(frame, totalLeftEyeBlinks, totalRightEyeBlinks, totalBothEyeBlinks, averageEAR):
    cv2.putText(frame, "Left Eye Blinks: {}".format(totalLeftEyeBlinks), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Right Eye Blinks: {}".format(totalRightEyeBlinks), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Both Eyes Blinks: {}".format(totalBothEyeBlinks), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Average EAR: {:.5f}".format(averageEAR), (290, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)