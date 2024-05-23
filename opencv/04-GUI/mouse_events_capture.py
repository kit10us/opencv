import sys
from pathlib import Path
import cv2
import numpy as np

windowName = "Window"
mouseDownPosition = (0, 0)
mouseUpPosition = (0, 0)
mouseIsDown = False
image = None
fileoutName = "captured.jpg"

def GetImagePath(filename) -> Path:
    return Path(sys.argv[0]).parent / filename


def callbackMouse(action, x, y, flags, userdata):
    global windowName
    global mouseDownPosition
    global mouseUpPosition
    global mouseIsDown
    global image

    workingImage = image.copy()
    if action == cv2.EVENT_LBUTTONDOWN:
        mouseIsDown = True
        mouseDownPosition = (x, y)
        mouseUpPosition = mouseDownPosition
    elif action == cv2.EVENT_LBUTTONUP and mouseIsDown == True:
        mouseIsDown = False
        cv2.imshow(windowName, image)
        workingImage = image[mouseDownPosition[1]:mouseUpPosition[1], mouseDownPosition[0]:mouseUpPosition[0]]
        cv2.imwrite(GetImagePath(fileoutName).as_posix(), workingImage)
        return
    elif action == cv2.EVENT_RBUTTONDOWN:
        mouseIsDown = False
    elif action == cv2.EVENT_MOUSEMOVE and mouseIsDown == True:
        mouseUpPosition = (x, y)
        rectangleColor = (0, 0, 255)
        rectangleThickness = 3
        rectangleLineType = cv2.LINE_AA
        workingImage = cv2.rectangle(workingImage, mouseDownPosition, mouseUpPosition, rectangleColor, rectangleThickness, rectangleLineType)
    else:
        return

    cv2.imshow(windowName, workingImage)

def main(argc: int, argv) -> int:
    global windowName
    global image

    imageFilename = GetImagePath("boy.jpg")

    image = cv2.imread(str(imageFilename))
    if image is None:
        raise FileNotFoundError("Could not load image file " + imageFilename)

    # Write text
    text = "Press any key to exit, LB to copy, RB to cancel"
    imageHeight, imageWidth = image.shape[0:2]
    textFontFace = cv2.FONT_HERSHEY_PLAIN
    textScale = 1
    textColor = (255, 255, 255)
    textThickness = 1
    textLineType = cv2.LINE_AA
    textSize = cv2.getTextSize(text, textFontFace, textScale, textThickness)[0]
    textPosition = (int(imageWidth / 2 - textSize[0] / 2), int(imageHeight - 2 * textSize[1]))
    image = cv2.putText(image, text, (textPosition[0] + 2, textPosition[1] + 2), textFontFace, textScale, (0, 0, 0), textThickness, textLineType)
    image = cv2.putText(image, text, textPosition, textFontFace, textScale, textColor, textThickness, textLineType)

    cv2.imshow(windowName, image)
    cv2.setMouseCallback(windowName, callbackMouse)

    isDone = False
    while isDone == False:
        isDone = cv2.waitKey(1) != -1 or int(cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) == 0)
        

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))






