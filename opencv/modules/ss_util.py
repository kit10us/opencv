import sys
from enum import Enum
from pathlib import Path


def TimeNSToMicroS(timeNS) -> float:
    return timeNS * 0.001

def TimeNSToMS(timeNS) -> float:
    return TimeNSToMicroS(timeNS) * 0.001

def TimeNSToS(timeNS) -> float:
    return TimeNSToMS(timeNS) * 0.001

class AssetType(Enum):
    Image = 0
    Video = 1

def GetAssetPath(filename: str, assetType: AssetType, rootPath: Path = None) -> Path:
    """
    Gets an asset's path based on the asset type.
    The current path is always checked first. Then the list of paths relative to the root.

    Args:
        filename (str):         The name of the file to get the path of.
        assetType (AssetType):  The type of the asset to get (it is an enum).
        rootPath (Path):        The root path to use when checking. None uses the current path as the root.

    Returns:
        Path: The path to the asset.
    """
    testPath: Path = None
    if rootPath is None:
        rootPath = Path(sys.argv[0]).parent
    testPath = rootPath / Path(filename)
    if testPath.exists():
        return testPath
    assetRootPath: Path = None
    match assetType:
        case AssetType.Image:
            assetRootPath = Path("images")
        case AssetType.Video:
            assetRootPath = Path("videos")
    testPath = rootPath.parent / assetRootPath / filename
    if testPath.exists():
        return testPath
    raise FileNotFoundError(f"Asset file {filename} not found.")
