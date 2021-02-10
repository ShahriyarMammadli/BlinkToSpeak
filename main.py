# Shahriyar Mammadli
# Import required libraries.
import helperFunctions as hf
import imutils
from imutils import face_utils
import cv2
import dlib
import morseDecoder as md

# Set the parameters.
# Define the face landmark path.
landmarkPath = "shape_predictor_68_face_landmarks.dat"
# Threshold value for EAR.
EARThreshold = 0.225
#  The threshold for number of consecutive frames to wait before...
#  ...validating eye blink.
conFrames = 3
# Counter of current number of frames that eye-blink has been continuing.
curLeftEyeBlinkFrames = 0
curRightEyeBlinkFrames = 0
curBothEyeBlinkFrames = 0
# Total number of eye blinks.
totalLeftEyeBlinks = 0
totalRightEyeBlinks = 0
totalBothEyeBlinks = 0
# Set the frame width.
frameWidth = 450
# The face detector of dlib is used to detect faces in a frame.
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector initialization.
predictor = dlib.shape_predictor(landmarkPath)
# Indices of the landmarks of each eye.
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# Initialization of video stream.
videoStream = imutils.video.VideoStream(src=0).start()

# Message content will be hold in this variable
message = ""

# Run the system while 'q' is not pressed.
while True:
    # Obtain the frames from the camera steam.
    frame = videoStream.read()
    # Resize the image.
    frame = imutils.resize(frame, width=frameWidth)
    # Convert the image to grayscale before detecting landmarks.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces in the frame.
    faces = faceDetector(gray, 0)

    # Iterate over the faces.
    for (index, rect) in enumerate(faces):
        # Get the landmarks of a face
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Select the landmarks of left and right eyes.
        leftEye = shape[leftStart:leftEnd]
        rightEye = shape[rightStart:rightEnd]
        leftEAR = hf.EAR(leftEye)
        rightEAR = hf.EAR(rightEye)
        # average the eye aspect ratio together for both eyes.
        averageEAR = (leftEAR + rightEAR) / 2.0
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # ear_ratio = ear / w
        # Draw the details on the frame.
        hf.drawOnStream(frame, rect, index, shape, leftEye, rightEye)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if leftEAR < EARThreshold and rightEAR < EARThreshold and abs(leftEAR - rightEAR) < 0.015:
            curBothEyeBlinkFrames += 1
            curLeftEyeBlinkFrames += 1
            curRightEyeBlinkFrames += 1
        elif leftEAR < EARThreshold and rightEAR - leftEAR > 0.015:
            curLeftEyeBlinkFrames += 1
        elif rightEAR < EARThreshold and leftEAR - rightEAR > 0.015:
            curRightEyeBlinkFrames += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold:
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if curLeftEyeBlinkFrames >= conFrames:
                totalLeftEyeBlinks += 1
                print(leftEAR)
                print(rightEAR)
            if curRightEyeBlinkFrames >= conFrames:
                totalRightEyeBlinks += 1
                print(leftEAR)
                print(rightEAR)
            if curLeftEyeBlinkFrames >= conFrames and curRightEyeBlinkFrames >= conFrames:
                totalBothEyeBlinks += 1
            # Build message content
            if curLeftEyeBlinkFrames >= conFrames > curRightEyeBlinkFrames:
                message += '.'
            elif curRightEyeBlinkFrames >= conFrames > curLeftEyeBlinkFrames:
                message += '-'
            elif curLeftEyeBlinkFrames >= conFrames and curRightEyeBlinkFrames >= conFrames:
                message += ' '

            print(message)
            decryptedMessage = md.decrypt(message)
            print(decryptedMessage)
            # reset the eye frame counter
            curLeftEyeBlinkFrames = 0
            curRightEyeBlinkFrames = 0
            curBothEyeBlinkFrames = 0
        # Draw results.
        hf.drawResults(frame, totalLeftEyeBlinks, totalRightEyeBlinks, totalBothEyeBlinks, averageEAR)
    cv2.imshow("BlinkToSpeak", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Shut off the stream.
cv2.destroyAllWindows()
videoStream.stop()