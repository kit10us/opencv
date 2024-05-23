import sys
import cv2 
from pathlib import Path

def main(argc: int, argv) -> int:
    imagesPath = Path(argv[0]).parent.parent / Path("images")
    imageFilePath =  imagesPath / Path("amsterdam_bikes.jpg")
    img = cv2.imread(imageFilePath.as_posix())
    if img is None:
        raise FileNotFoundError("Unable to open image " + imageFilePath.as_posix())
    
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    # Apply a color scale to the image.
    for y in range(imgHeight):
        for x in range(imgWidth):
            color = img[y][x]
            sc = (x / imgWidth + y / imgHeight) * 0.5            
            color = color * (1.0 - sc, 1.0 - sc, 1.0 - sc)
            img[y][x] = color

    # Write image dimensions to the window
    imageText = "size is " + str(imgWidth) + " by " + str(imgHeight)
    fontType = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    textThickness = 1
    textSize = cv2.getTextSize(imageText, fontType, fontScale, textThickness)
    textWidth = textSize[0][0]
    textHeight = textSize[0][1]
    textOrg = (
        int(imgWidth / 2 - textWidth / 2),
        int(imgHeight - textHeight * 2)
        )
    textColor = (255, 255, 255)    
    cv2.putText(img, imageText, (textOrg[0] + 2, textOrg[1]  + 2), fontType, fontScale, (0,0,0), textThickness, cv2.LINE_AA)
    cv2.putText(img, imageText, textOrg, fontType, fontScale, textColor, textThickness, cv2.LINE_AA)

    # Show image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))
