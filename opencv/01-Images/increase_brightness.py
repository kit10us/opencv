import sys
import cv2
import numpy as np
from pathlib import Path

def main(argc: int, argv) -> int:
    def GetPath(filenameIn):
        return (Path(argv[0]).parent.parent / Path("images") / Path(filenameIn)).as_posix()
    
    image = cv2.imread(GetPath("amsterdam_bikes.jpg"))

    constrastIncrease = 60
    image = cv2.add(image, constrastIncrease)

    windowName = "Image"
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    floatImage = np.float32(image) / 255
    

    cv2.destroyAllWindows()    
    
    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))