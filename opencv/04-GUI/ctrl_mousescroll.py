import sys
import cv2
from pathlib import Path

def GetImagePath(filename) -> Path:
    return Path(sys.argv[0]).parent.parent / "images" / filename

imagePresentation: cv2.Mat = None


def main(argc: int, argv : list[str]) -> int:
    imagePath = GetImagePath("threshold.png")
    imageOriginal = cv2.imread(imagePath.as_posix())
    if imageOriginal is None:
        raise Exception(f"Could not load {str(imagePath)}")

    windowName = "Window"
    
    imagePresentation = imageOriginal.copy()

    scalingFactor: float = [1.0]
    scalingType: any = None

    def resizeImage(action, x, y, flag, userdata):
        # Referencing global variables 
        answer_1 = cv2.EVENT_MOUSEWHEEL
        answer_2 = cv2.EVENT_FLAG_CTRLKEY
        answer_3 = flag
        global imagePresentation
        #global scalingFactor, scalingType
        if (action == answer_1) and (flag & answer_2):
            # Action to be taken when ctrl key + mouse wheel scrolled forward
            if (answer_3 > 0):
                scalingFactor = userdata[0][0]
                scalingFactor += 0.1
                if scalingFactor > 2.0:
                    scalingFactor = 2.0
                userdata[0][0] = scalingFactor
                imagePresentation = cv2.resize(imageOriginal, None, None, float(scalingFactor), float(scalingFactor))
            # Action to be taken when ctrl key + mouse wheel scrolled backward
            elif answer_3 < 0:
                scalingFactor = userdata[0][0]
                scalingFactor -= 0.1
                if scalingFactor < 0.1:
                    scalingFactor = 0.1
                userdata[0][0] = scalingFactor
                imagePresentation = cv2.resize(imageOriginal, None, None, float(scalingFactor), float(scalingFactor))
                # Resize image    


    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, resizeImage, [scalingFactor])
    k = -1
    while(k != 113):
        imagePresentation = cv2.resize(imageOriginal, None, None, float(scalingFactor[0]), float(scalingFactor[0]))
        cv2.imshow(windowName, imagePresentation)
        k = cv2.waitKeyEx(25)
    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))