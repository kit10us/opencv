import cv2
import numpy as np

def GenerateHistogram(source: cv2.Mat) -> np.ndarray:
    histogram: list[np.ndarray] = np.zeros([3, 256], np.uint8)
    for y in range(0, source.shape[0]):
        for x in range(0, source.shape[1]):
            pixel = source[y, x]
            red = pixel[2]
            green = pixel[1]
            blue = pixel[0]
            histogram[0][red] += 1
            histogram[1][green] += 1
            histogram[2][blue] += 1
    return histogram

def CreateImageFromHistogram(channel: np.ndarray, width: int, height: int, color: list[3] = None, backgroundColor: list[3] = None) -> cv2.Mat:
    if color is None:
        color = [255, 255, 255]
    if backgroundColor is None:
        backgroundColor = [0, 0, 0]

    maxPixels: int = channel.max()
    
    outputWidth = 256
    outputHeight = 256
    output: cv2.Mat = np.zeros([outputHeight, outputWidth, 3], np.uint8)
    thickness: int = 1
    output = cv2.rectangle(output, (0, 0), (output.shape[1], output.shape[0]), backgroundColor, cv2.FILLED)
    for x in range(0, len(channel)):
        count: int = channel[x]
        lineUL = (x, outputHeight - 1)
        lineDR = (x, outputHeight - 1 - count)
        cv2.line(output, lineUL, lineDR, color, thickness, cv2.LINE_AA)

    output = cv2.resize(output, (width, height))
    return output