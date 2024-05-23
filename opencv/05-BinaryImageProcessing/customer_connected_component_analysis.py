import sys
from pathlib import Path
import cv2
import numpy as np

class MinMax:
    min:int = 0
    max:int = 0

    def __init__(self, min: int, max: int):
        if min > max:
            assert "Min is greater than max"

        self.min = min
        self.max = max

    def __len__(self) -> int:
        return self.max - self.min
    
    def __iter__(self):
        self._i = self.min
        return self
    
    def __next__(self):
        if self._i > self.max:
            raise StopIteration
        i = self._i
        self._i += 1
        return i

def GetImagePath(filename) -> Path:
    return Path(sys.argv[0]).parent.parent / "images" / filename

# Returns the lowest connectivity (for current)
def ConnectedComponents(source: cv2.Mat):
    # Ensure all blobs connect to lowest connect blob
    # Always return the lowest connection
    # We stop when either current's or adjacent's parent is not connected (0)
    # We want current to always be the lowest
    def ConnectBlob(blobConnectivity, current: int, adjacent: int):
        if current == 0 or adjacent == 0:
            return current
        if current == adjacent:
            return current
        
        current_parent = blobConnectivity[current]
        adjacent_parent = blobConnectivity[adjacent]

        if current == adjacent:
            return current
        elif current < adjacent: # 1, 2, [0, 0] = 0, 1, [0, 1]
            if current_parent >= adjacent:
                current_parent = adjacent
            


                blobConnectivity[current] = adjacent_parent
                blobConnectivity[adjacent] = current
                return current
            
        

            if adjacent_parent == current:
                return current, current
            if adjacent_parent > current:
                blobConnectivity[adjacent] = ConnectBlob(blobConnectivity, current, adjacent_parent)
                return blobConnectivity[adjacent]
            if adjacent_parent < current:
                blobConnectivity[current] = ConnectBlob(blobConnectivity, adjacent_parent, current)
                return blobConnectivity[current]
        elif current > adjacent:
            adjacent, blobConnectivity[current] = ConnectBlob(blobConnectivity, adjacent, current)
            return adjacent, adjacent
            
        raise Exception("ConnectBlob went too far")
    
    def UnitTest():
        blobConnectivity = np.arange(256)

        print("START")
        print(1)
        blobConnectivity[1] = 0
        blobConnectivity[2] = 0
        current, adjacent = 0, 0
        print("A"); assert ConnectBlob(blobConnectivity, current, adjacent ) == 0,  "Failed current = " + str(current) + ", adjacent = " + str(adjacent)
        print("B"); assert(blobConnectivity[1] == 0 )
        print("C"); assert(blobConnectivity[2] == 0 )
        print("")
        
        print(2)
        blobConnectivity[1] = 0
        blobConnectivity[2] = 0
        current, adjacent = 1, 0
        print("A"); assert ConnectBlob(blobConnectivity, current, adjacent ) == 1,  "Failed"
        print("B"); assert(blobConnectivity[1] == 0 )
        print("C"); assert(blobConnectivity[2] == 0 )
        print("")
        
        print(3)
        blobConnectivity[1] = 0
        blobConnectivity[2] = 0
        current, adjacent = 0, 1
        print("A"); assert ConnectBlob(blobConnectivity, current, adjacent ) == 0,  "Failed"
        print("B"); assert(blobConnectivity[1] == 0 )
        print("C"); assert(blobConnectivity[2] == blobConnectivity[2] )
        print("")
        
        print(4)
        blobConnectivity[1] = 0
        blobConnectivity[2] = 0
        current, adjacent = 1, 1
        print("A"); assert ConnectBlob(blobConnectivity, current, adjacent ) == 1,  "Failed"
        print("B"); assert(blobConnectivity[1] == 0 )
        print("C"); assert(blobConnectivity[2] == 0 )
        print("")

        blobConnectivity[1] = 0
        blobConnectivity[2] = 0
        print(5)
        current, adjacent = 1, 2
        print("A"); assert ConnectBlob(blobConnectivity, current, adjacent ) == 0,  "Failed"
        print("B"); assert(blobConnectivity[1] == 0 )
        print("C"); assert(blobConnectivity[2] == current )
        print("")

        blobConnectivity[1] = 2
        blobConnectivity[2] = 0
        print(6)
        current, adjacent = 1, 2
        print("A"); assert ConnectBlob(blobConnectivity, current, adjacent ) == 1,  "Failed"
        print("B"); assert(blobConnectivity[1] == 0 )
        print("C"); assert(blobConnectivity[2] == current )
        print("")

        print("PASSED")
    UnitTest()
    exit(0)


    labelImage = np.zeros_like(source) # 3d hash
    height, width = source.shape[0:2]
    blobConnectivity = np.arange(256)

    # There are only four adjacents to check
    adjacents_list = [[-1, -1], [0, -1], [1, -1], [-1, 0]]

    # Connectivity 0 is reserved for background.
    current_label = 1
    blobConnectivity[1] = current_label
    lowest_label = current_label
    label_used: bool = False

    UnitTest()


    for y in range(height):
        for x in range(width):
            if source[y, x] == 0: # if current pixel is false, continue
                if label_used:
                    label_used = False
                    current_label += 1
                    print(current_label)
            else:
                label_used = True
                any_adjacent:bool = False
                lowest_label: int = current_label
                adjacent_label: int = 0
                for adjacent in adjacents_list:
                    adjx, adjy = (x + adjacent[0], y + adjacent[1])
                    if adjx < 0 or adjx >= width or adjy < 0 or adjy >= height or source[adjy, adjx] == 0 or source[adjy, adjx] == current_label:
                        continue # if adjacent is out of bounds, false, or the same as us
                    else:
                        lowest_label = ConnectBlob(blobConnectivity, current_label, labelImage[adjy, adjx])
                    
                labelImage[y, x] = lowest_label
                current_label = lowest_label

    # Ensure we are using the lowest connectivities
    for y in range(height):
        for x in range(width):
            labelImage[y, x] = blobConnectivity[labelImage[y, x]]
    
    return current + 1, blobConnectivity

def main(argc: int, argv) -> int:
    filePath = GetImagePath("truth.png")
    image = cv2.imread(filePath.as_posix(), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Failed to load image " + image)
    
    connectedImage = ConnectedComponents(image)


    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))