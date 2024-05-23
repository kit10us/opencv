import sys
import cv2
from pathlib import Path

def main(argc: int, argv) -> int:
    imagesPath = Path(argv[0]).parent.parent / Path("images")
    imageFilePath =  imagesPath / Path("amsterdam_bikes.jpg")
    img = cv2.imread(imageFilePath.as_posix())
    if img is None:
        raise FileNotFoundError("Unable to open image " + imageFilePath.as_posix())
    
    windowName = "Video"
    cv2.imshow(windowName, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))