import sys
import cv2
from pathlib import Path

def main(argc: int, argv) -> int:
    imagesPath = Path(argv[0]).parent.parent / Path("images")
    imageFilePath =  imagesPath / Path("amsterdam_bikes.jpg")
    img = cv2.imread(imageFilePath.as_posix())
    if img is None:
        raise FileNotFoundError("Unable to open image " + imageFilePath.as_posix())

    imageText = "size is " + str(img.shape[0]) + " by " + str(img.shape[1])
    fontType = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    textSize = cv2.getTextSize(imageText, fontType, 1, 1)
    textThickness = 1
    textOrg = (
        int(img.shape[1] / 2 - textSize[0][0] / 2),
        int(img.shape[0] - textSize[0][1] * 2)
        )
    print(type(textOrg[0]))
    textColor = (255, 255, 255)
    cv2.putText(img, imageText, (textOrg[0] + 2, textOrg[1]  + 2), fontType, fontScale, (0,0,0), textThickness, cv2.LINE_AA)
    cv2.putText(img, imageText, textOrg, fontType, fontScale, textColor, textThickness, cv2.LINE_AA)

    # Show image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))
