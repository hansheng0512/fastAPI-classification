import torchvision.transforms as transforms

TRANSFORM = transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

NUM_CLASSES = 1000

with open('classifier/imagenet_classes.txt') as f:
    CLASSES = []
    for line in f.readlines():
        line = line.split(", ")[1].strip()
        CLASSES.append(line)