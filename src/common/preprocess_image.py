
from torchvision import transforms

_preprocess = transforms.Compose([
    transforms.ToTensor()
])

def preprocess(image):
    """
    Converts an image to a GPU tensor and subtracts -0.5
    :return: a tensor image with value ranging from -0.5 to 0.5
    """
    as_tensor = _preprocess(image).cuda()
    as_tensor -= 0.5
    return as_tensor
