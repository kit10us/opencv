import sys
import cv2
import numpy as np
from  pathlib import Path
import types



def main(argc: int, argv) -> int:
    def GetPath(filenameIn):
        return (Path(argv[0]).parent.parent / Path("images") / Path(filenameIn)).as_posix()
    
    windowName = "Image"
    
    # Create image
    newMatrix = np.zeros((1000,1000, 3), dtype = "uint8")
    cv2.imshow("Image", newMatrix)
    cv2.waitKey(0)

    newMatrix = 255 * np.ones((1000,1000, 3), dtype = "uint8")
    cv2.imshow("Image", newMatrix)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))