import sys
import cv2
import numpy as np
from pathlib import Path

def main(argc: int, argv) -> int:
    def GetPath(source: str) -> str:
        return Path(Path(argv[0]).parent.parent / Path("images") / Path(source)).as_posix()
    
    backgroundImage = cv2.imread(GetPath("amsterdam_bikes.jpg"))
    backgroundImage = np.float32(backgroundImage) / 255
    backgroundHeight, backgroundWidth = backgroundImage.shape[0:2]
    
    overlayImage = cv2.imread(GetPath("transparent_boo.png"), cv2.IMREAD_UNCHANGED)
    #overlayImage = cv2.imread(GetPath("transparent_butterly.png"), cv2.IMREAD_UNCHANGED)
    overlayImage = np.float32(overlayImage) / 255
    
    overlayHeight, overlayWidth = overlayImage.shape[0:2]
    overlayImage = cv2.resize(overlayImage, None, overlayImage, backgroundHeight / overlayHeight * 0.15, backgroundWidth / overlayWidth * 0.15, cv2.INTER_LINEAR)
    overlayHeight, overlayWidth = overlayImage.shape[0:2]

    destX = 300
    destY = 80

    overlayBGR = overlayImage[:,:,0:3]
    overlayMask1Channel = overlayImage[:,:,3:4]
    overlayMask = cv2.merge((overlayMask1Channel, overlayMask1Channel, overlayMask1Channel))
    overlayMasked = cv2.multiply(overlayBGR, overlayMask)

    backgroundROI = backgroundImage[destY:destY + overlayHeight, destX:destX + overlayWidth]
    backgroundROIMasked = cv2.multiply(backgroundROI, (1 - overlayMask), None, 1)
    
    backgroundROIFinal = cv2.add(backgroundROIMasked, overlayMasked)
    
    finalImage = np.zeros_like(backgroundImage)
    np.copyto(finalImage, backgroundImage)
    finalImage[destY:destY + overlayHeight, destX:destX + overlayWidth] = backgroundROIFinal

    windowName = "Image"
    cv2.imshow(windowName, finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))