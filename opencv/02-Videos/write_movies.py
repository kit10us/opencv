import sys
import cv2
from pathlib import Path
import time
import numpy as np

def GetVideoPath(filename: str) -> Path:
     return Path(sys.argv[0]).parent.parent / "videos" / filename

def GetOutputPath(filename: str) -> Path:
     return Path(sys.argv[0]).parent.parent / "output" / filename

def DrawWindow(window: str, image: cv2.Mat, text: str):
    imageHeight, imageWidth = image.shape[0:2]
    textFontFace = cv2.FONT_HERSHEY_PLAIN
    textScale = 1
    textColor = (255, 255, 255)
    textThickness = 1
    textLineType = cv2.LINE_AA
    textSize = cv2.getTextSize(text, textFontFace, textScale, textThickness)[0]
    textPosition = (int(imageWidth / 2 - textSize[0] / 2), int(imageHeight - 2 * textSize[1]))
    drawWindow = image[:,:]
    drawWindow = cv2.putText(drawWindow, text, (textPosition[0] + 2, textPosition[1] + 2), textFontFace, textScale, (0, 0, 0), textThickness, textLineType)
    drawWindow = cv2.putText(drawWindow, text, textPosition, textFontFace, textScale, textColor, textThickness, textLineType)
    cv2.imshow(window, drawWindow)


def main(argc: int, argv) -> int:
    videoFilePath = GetVideoPath("cars_overhead01.mp4")

    videoReader = cv2.VideoCapture(videoFilePath.as_posix())
    if not videoReader.isOpened():
        raise FileExistsError("Failed to load video" + videoFilePath.as_posix())

    movieSize = int(videoReader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    movieFrames = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    movieFPS = int(videoReader.get(cv2.CAP_PROP_FPS))
    videoWindow = "Video"

    videoWriter = None
    outFilename = GetOutputPath("movie_upsidedown.mp4")
    videoWriter = cv2.VideoWriter(outFilename.as_posix(), cv2.VideoWriter.fourcc(*'avc1'), movieFPS, movieSize)

    i = 0
    while True:
        i += 1
        readSuccess, imageFrame = videoReader.read()
        if not readSuccess:
            break
        
        imageFrame = cv2.flip(imageFrame, 0)
        imageFrame = cv2.flip(imageFrame, 1)
        videoWriter.write(imageFrame)

        DrawWindow(videoWindow, imageFrame, "Frame: " + str(i))
        cv2.waitKey(movieFPS)
        if cv2.getWindowProperty(videoWindow, cv2.WND_PROP_VISIBLE) < 1:
            break

    videoWriter.release()
    videoReader.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))