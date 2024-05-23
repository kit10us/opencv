import numpy as np

class Rect:
    __ul: list[int] = 0, 0
    __dr: list[int] = 0, 0

    def __init__(self, ul: list[int], dr: list[int]):
        self.__ul = ul
        self.__dr = dr

    def getUL(self) -> list[int]:
        """
        Returns:
            ul (list[int]): [0] = x, [1] = y
        """
        return self.__ul
    
    def setUL(self, ul: list[int]):
        self.__ul = ul

    def getDR(self) -> list[int]:
        """
        Returns:
            dr (list[int]): [0] = x, [1] = y
        """
        return self.__dr
    
    def setDR(self, dr: list[int]):
        self.__dr = dr

    def getU(self) -> int:
        return self.__ul[1]

    def getL(self) -> int:
        return self.__ul[0]

    def getD(self) -> int:
        return self.__dr[1]

    def getR(self) -> int:
        return self.__dr[0]

    def setU(self, u: int):
        self.__ul[1] = u

    def setL(self, l: int):
        self.__ul[0] = l

    def setD(self, d: int):
        self.__dr[1] = d

    def setR(self, r: int):
        self.__dr[0] = r

    def get2D(self) -> list[list[int]]:
        return self.__ul[:] + self.__dr[:]
    
    def get3D(self) -> list[list[int]]:
        return [self.__ul[:], self.__dr[:]]
    
    def getWidth(self) -> int:
        return (self.r - self.l) + 1

    def getHeight(self) -> int:
        return (self.d - self.u) + 1

    ul = property(getUL, setUL)
    dr = property(getDR, setDR)

    u = property(getU, setU)
    l = property(getL, setL)

    d = property(getD, setD)
    r = property(getR, setR)

    array_2d = property(get2D)
    array_3d = property(get3D)

    width = property(getWidth)
    height = property(getHeight)

    def crop(self, array: np.ndarray) -> np.ndarray:
        cropped: np.ndarray = np.zeros([self.height, self.width, array.shape[2]], np.uint8)
        cropped[:,:] = array[self.u:self.d + 1, self.l:self.r + 1, :]
        return cropped
    
    def paste(self, array_in: np.ndarray, array_out: np.ndarray) -> None:
        array_out[self.u:self.d + 1, self.l:self.r + 1, :] = array_in[0:self.height, 0:self.width, :]