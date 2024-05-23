import sys
import cv2
from pathlib import Path
import time

def main(argc: int, argv) -> int:
    videosPath = Path(argv[0]).parent.parent / Path("videos")
    videoFilePath = videosPath / Path("chaplin.mp4")

    capture = cv2.VideoCapture(videoFilePath.as_posix())
    print(capture.get(cv2.CAP_PROP_FOURCC))
    print(cv2.VideoWriter.fourcc(*"h264"))
    if not capture.isOpened():
        raise FileExistsError("Failed to load video" + videoFilePath.as_posix())

    videoWindow = "Video"

    # Show video at FPS, taking in account the time a frame took to show the frame.
    showTimeLengthMS = 0
    while True:
            showTimeStartNS = time.time_ns()

            readSuccess, imageFrame = capture.read()
            if not readSuccess:
                 break
                 

            cv2.imshow(videoWindow, imageFrame)

            cv2.waitKey(int(1000 / (capture.get(cv2.CAP_PROP_FPS) - showTimeLengthMS)))
            if cv2.getWindowProperty(videoWindow, cv2.WND_PROP_VISIBLE) < 1:
                break

            showTimeEndNS = time.time_ns()
            showTimeLengthMS = 1000000 / (showTimeEndNS - showTimeStartNS)

    capture.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))