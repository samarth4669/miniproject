import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage.color import rgb2gray
import os
from scipy import spatial



# Function to grab frames from a video file
def frame_grabber(file):
    frames = []
    cap = cv2.VideoCapture(file)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    frames = np.asarray(frames)
    return frames

# Function to read image files from a directory
def read_files(file):
    filename = glob.glob(file + '*png')
    file_name = {}
    for i in range(len(filename)):
        file_name[i] = np.double(Image.open(str(filename[i])).convert('L'))
    return file_name

# Function for background subtraction
def background_subtraction(Im, background, threshold):
    bgs = {}
    for i in range(len(Im)):
        bgs[i] = (np.abs(Im[i] - background) > threshold).astype(int)
    return bgs

# Function to calculate Motion History Image (MHI)
def MHI(image, delta):
    mhi = np.zeros((np.shape(image[0])[0], np.shape(image[0])[1]), np.uint8)
    row, column = np.shape(image[0])
    for timestamp in range(0, len(image)):
        frame = image[timestamp]
        for y in range(row):
            for x in range(column):
                if (frame[y, x] == 1):
                    mhi[y, x] = timestamp + 1
                else:
                    if (mhi[y, x] < timestamp - delta):
                        mhi[y, x] = 0
    return mhi

# Function to calculate Motion Energy Image (MEI)
def MEI(Im):
    mei = np.zeros((np.shape(Im[0])[0], np.shape(Im[0])[1]), np.uint8)
    for i in range(len(Im)):
        mei = mei + Im[i]
        mei = mei > 0
    return np.asarray(mei)

# Function to calculate MEI using thresholded MHI
def MEI_Thresh(mhi):
    mei = mhi > 0
    return mei

# Function to normalize MHI
def normalize(mhi):
    mhi_n = np.maximum(0, np.divide((mhi - (np.min(mhi[np.nonzero(mhi)]) - 1.0)),(np.max(mhi[np.nonzero(mhi)]) - (np.min(mhi[np.nonzero(mhi)]) - 1.0))))
    return mhi_n

# Function to calculate similitude moments
def similitude_moments(Im):
    y, x = np.mgrid[range(Im.shape[0]), range(Im.shape[1])]
    similitude_moments = []
    x_bar = np.sum(x * Im) / np.sum(Im)
    y_bar = np.sum(y * Im) / np.sum(Im)
    for i in range(4):
        for j in range(4):
            if (2 <= (i + j) <= 3):
                s = np.sum(((x - x_bar) ** i) * ((y - y_bar) ** j) * Im) / (np.sum(Im)) ** (((i + j) / 2) + 1)
                similitude_moments.append(s)
    return similitude_moments

# Function to get temporal template
def get_temporal_template(file_name, bg_file_name):
    thresh = 40
    delta = 30
    imgFrameData = frame_grabber(file_name)
    imgGrayscaleFrameData = {}
    for i, image in enumerate(imgFrameData):
        grayscale = rgb2gray(image)
        imgGrayscaleFrameData[i] = (grayscale * 255).astype(int)
    bgImage = (rgb2gray(bg_file_name) * 255).astype(int)
    bgSubsImages = background_subtraction(imgGrayscaleFrameData, bgImage, threshold=thresh)
    mhiImg = MHI(bgSubsImages, delta)
    normMhiImg = normalize(mhiImg)
    meiImg = MEI_Thresh(normMhiImg)
    mhiSimilitude = similitude_moments(normMhiImg)
    meiSimilitude = similitude_moments(meiImg)
    result = []
    result.extend(mhiSimilitude)
    result.extend(meiSimilitude)
    return result

# Function to extract background from a video file
def extract_background(video_file):
    cap = cv2.VideoCapture(video_file)
    _, first_frame = cap.read()
    background_model = first_frame.astype(float)
    num_frames = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_float = frame.astype(float)
        background_model = (background_model * num_frames + frame_float) / (num_frames + 1)
        num_frames += 1
    cap.release()
    background_image = background_model.astype('uint8')
    return background_image

# Video file path
t = r'/content/drive/MyDrive/Diving-Side/Walk-Front/022/7608-3_70626.avi'
# Extract background from video
bg1 = extract_background(t)
# Display background image
plt.imshow(cv2.cvtColor(bg1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Get temporal template
out1 = get_temporal_template(t, bg1)
