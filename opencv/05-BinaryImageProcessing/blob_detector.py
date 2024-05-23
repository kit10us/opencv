import sys
import cv2

from ss_util import GetAssetPath
from ss_util import AssetType
from ss_showimages import ImageList
from ss_showimages import ShowImages

def main(argc: int, argv: list[str]) -> int:
    
    def DrawBlobs(sourceImage, keypoints) -> cv2.Mat:
        resultImage = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        for point in keypoints:
            x, y = int(round(point.pt[0])), int(round(point.pt[1]))
            cv2.circle(resultImage, (x, y), 5, (255, 0, 0), -1)        
            cv2.circle(resultImage, (x, y), int(point.size / 2), (255, 0, 0), 2)
        return resultImage


    image1Path = GetAssetPath("5_new.png", AssetType.Image)
    image1 = cv2.imread(image1Path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if image1 is None:
        raise Exception("File " + image1Path.as_posix())

    image2Path = GetAssetPath("6_new.png", AssetType.Image)
    image2 = cv2.imread(image2Path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if image2 is None:
        raise Exception("File " + image2Path.as_posix())
    
    
    sbd_params = cv2.SimpleBlobDetector.Params()
    sbd_params.filterByArea = False
    sbd = cv2.SimpleBlobDetector.create(parameters=sbd_params)
    
    imageList: ImageList = ImageList(1500, 5, 30)
    imageList.append("Original Image 1", image1)
    _, image1Binary = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    image1Binary = cv2.cvtColor(image1Binary, cv2.COLOR_GRAY2BGR)
    image1Keypoints = sbd.detect(image1Binary)
    image1Blobs = DrawBlobs(image1, image1Keypoints)
    imageList.append(f"Blobs Image 1 ({len(image1Keypoints)})", image1Blobs)

    imageList.append("Original Image 2", image2)
    _, image2Binary = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
    image2Binary = cv2.cvtColor(image2Binary, cv2.COLOR_GRAY2BGR)
    image2Keypoints = sbd.detect(image2Binary)
    image2Blobs = DrawBlobs(image2, image2Keypoints)
    imageList.append(f"Blobs Image 2 ({len(image2Keypoints)})", image2Blobs)

    windowName = "Window"

    ShowImages(windowName, imageList)

    return 0


if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))


