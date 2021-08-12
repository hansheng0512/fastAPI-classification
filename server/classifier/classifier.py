from torchvision import models
import torch
from torch import nn

from classifier.config import TRANSFORM, CLASSES

class AlexNet:
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)

    def predict(self, image):
        image = TRANSFORM(image)
        image = image.unsqueeze(0)
        output = self.model(image)
        class_index = torch.argmax(output, dim=1)
        return CLASSES[class_index]

if __name__ == "__main__":
    model = AlexNet()
    print(model)
