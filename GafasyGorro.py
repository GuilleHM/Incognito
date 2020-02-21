'''
Script for recording a video, recognizing a face on it and make it become
incognito by overlapping a hat and sunglasses, using OpenCV library for it.
'''

import numpy as np
import cv2

# Webcam capture object instantiation
cap = cv2.VideoCapture(0)

# Video output configuration
save_path = 'multimedia/gafasygorro.mp4'
frames_per_seconds = 6
dims = (960, 540)
cap.set(3, dims[0])
cap.set(4, dims[1])
video_type = cv2.VideoWriter_fourcc(*'XVID') # video Encoding
out = cv2.VideoWriter(save_path, video_type, frames_per_seconds, dims)

# Image sections (face and eyes) classifiers instantiation
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_righteye_2splits.xml')

# Pictures for "incognito mode" :)
glasses = cv2.imread("misc/fun/rayban.png", -1)
hat = cv2.imread('misc/fun/hat.png',-1)

# Function for resizing glasses and hat pictures in order to adjust them to detected face and eyes sizes
# Source: https://stackoverflow.com/a/44659589
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # Initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # Check to see if the width is None
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # Otherwise, the height is None
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # Return the resized image
    return resized

def main():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Converting to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Face(-s) detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        # Conveting to 4-channels (Alpha included) color 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # If a face is detected, a hat is set on top of it
        # This script version works only for the first face found; it can be adjusted for working with more faces
        if len(faces) != 0:
            (x, y, w, h) = faces[0][0:4]
            roi_gray = gray[y:y+h, x:x+h]
            roi_color = frame[y:y+h, x:x+h]
            hat2 = image_resize(hat.copy(), width= w)
            hw, hh, hc = hat2.shape
            for i in range(0, hw):
                for j in range(0, hh):
                    if hat2[i, j][3] != 0: # alpha 0
                        frame[y - int(hh/1.4) + i, x + j] = hat2[i, j]

            # Detects eyes only if one face has been detected on this frame; then, draws sun glasses
            eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            if len(eyes) != 0:
                (ex, ey, ew, eh) = np.mean(np.array(eyes), axis = 0).astype(int)[0:4] # center picture among all detected eyes
                glasses2 = image_resize(glasses.copy(), width= int(w/1.1))
                gw, gh, gc = glasses2.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        if glasses2[i, j][3] != 0:
                            roi_color[ey + i - 10, int(w/17) + j] = glasses2[i, j] # -10 and w/17 offset to center them
                    
        # Back to 3-channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Save to video file
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()