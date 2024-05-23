import sys
import cv2
from pathlib import Path

def main(argc: int, argv) -> int:
    def GetPath(filenameIn):
        return (Path(argv[0]).parent.parent / Path("images") / Path(filenameIn)).as_posix()
    
    image = cv2.imread(GetPath("transparent_submarine.png"), cv2.IMREAD_UNCHANGED)
    assert image.size != 0, "Failed to load image"

    assert image.shape[2] == 4, "Image has to have four channels"
    imageWindow = "Image"
    fillColor = [90, 10, 10]
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            alphaFactor = image[y, x, 3] / 255
            invAlphaFactor = 1 - alphaFactor
            image[y, x,:] = [
                image[y, x, 0] + fillColor[0] * invAlphaFactor,
                image[y, x, 1] + fillColor[1] * invAlphaFactor,
                image[y, x, 2] + fillColor[2] * invAlphaFactor,
                0]

    cv2.imshow(imageWindow, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))