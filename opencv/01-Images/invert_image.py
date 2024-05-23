import sys
import cv2
from  pathlib import Path

def main(argc: int, argv) -> int:
    def GetPath(filenameIn):
        return (Path(argv[0]).parent.parent / Path("images") / Path(filenameIn)).as_posix()

    image = cv2.imread(GetPath("amsterdam_bikes.jpg"))
    image = image[:,:,:][::-1, ::-1, :]

    imageWindow = "Image"
    cv2.imshow(imageWindow, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))