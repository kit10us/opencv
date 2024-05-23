import sys
import cv2
import typing
from pathlib import Path
import tkinter as tk

def MSToSeconds(ms: float) -> float:
    return ms * 1000

def main(argc: int, argv: typing.Sequence[str]) -> int:
    def GetPath(filenameIn):
        return (Path(argv[0]).parent.parent / Path("images") / Path(filenameIn)).as_posix()
        
    imageFiles = ["amsterdam_bikes.jpg", "beer_glasses.jpg", "city_people.jpg"]
    windowName = "Slideshow"
    
    for filename in imageFiles:
        print(GetPath(filename))
        image = cv2.imread(GetPath(filename))
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]

        maxWidth = int(tk.Tk().winfo_screenwidth() / 3)
        maxHeight = int(tk.Tk().winfo_screenheight() / 3)

        if imageWidth > maxWidth or imageHeight > maxHeight:
            if imageWidth > imageHeight:
                factor = maxWidth / imageWidth
            else:
                factor = maxHeight / imageHeight
            
            imageWidth *= factor
            imageHeight *= factor

        image = cv2.resize(image, (int(imageWidth), int(imageHeight)))

        if image is None:
            raise FileNotFoundError("Unable to load file " + GetPath(filename))

        cv2.imshow(windowName, image)
        cv2.waitKey(MSToSeconds(5))
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))