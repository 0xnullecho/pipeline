from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

img = Image.open("./query-images-cropped/1.png")
plt.imshow(img)
plt.show()