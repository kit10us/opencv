#img for storing the "IDCard-Satya.png" image read.
#qrDecoder for storing the QRCodeDetector object.
#opencvData for storing the decoded text (present in the form of the QR Code).

# Outline
#   1. Read Image
#   2. Detect QR Code in the image
#   3. Draw bounding box around detected QR code
#   4. Print the decoded text
#   5. Save and display the result image

import sys
import cv2
from pathlib import Path

def main(argc: int, argv) -> int:
    print(cv2.__version__)

    def GetImagePath(filename) -> str:
        return (Path(argv[0]).parent.parent / Path("images") / Path(filename)).as_posix()

    def GetOutputPath(filename) -> str:
        return (Path(argv[0]).parent.parent / Path("output") / Path(filename)).as_posix()

    #imageFilename = "QRCode-Satya.png"
    imageFilename = "IDCard-Satya.png"
    img = cv2.imread(GetImagePath(imageFilename))
    if img is None:
        raise FileNotFoundError("Failed to open file " + imageFilename)
    
    imgFinal = img.copy()

    qcd = cv2.QRCodeDetector()

    qrText, bbox, rectifiedImage = qcd.detectAndDecode(imgFinal)

    # Draw bounding box
    print(bbox)
    bboxUL = (int(bbox[0][0][0]), int(bbox[0][0][1]))
    bboxDR = (int(bbox[0][2][0]), int(bbox[0][2][1]))
    imgFinal = cv2.rectangle(imgFinal, bboxUL, bboxDR, (255, 0, 255), 2, cv2.LINE_AA)

    # Resize the QR image
    imageWidth = img.shape[1] * 3
    imageHeight = img.shape[0] * 3
    imgFinal = cv2.resize(imgFinal, (imageWidth, imageHeight))

    fontType = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    fontThickness = 1
    textSize = cv2.getTextSize(qrText, fontType, fontScale, fontThickness)
    fontColor = (255,255,255)
    textPosition = (
        int(imageWidth / 2 - textSize[0][0] / 2),
        int(imageHeight - 2 * textSize[0][1])
        )

    cv2.putText(imgFinal, qrText, (textPosition[0] + 2, textPosition[1] + 2), fontType, fontScale, (0, 0, 0), fontThickness, cv2.LINE_AA)
    cv2.putText(imgFinal, qrText, textPosition, fontType, fontScale, fontColor, fontThickness, cv2.LINE_AA)

    outputFilename = "read_qrcode_output_image.png"
    cv2.imwrite(GetOutputPath(outputFilename), imgFinal)

    windowName = "Image"
    cv2.imshow(windowName, imgFinal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))