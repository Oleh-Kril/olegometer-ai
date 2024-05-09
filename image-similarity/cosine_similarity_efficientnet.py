from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
from torch import nn

model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()

# Load the model
def get_similarity(
    image1_path = "a.jpeg",
    image2_path = "b-color-diff.jpeg"
):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Resize the smaller image to match the dimensions of the larger one
    max_width = max(image1.width, image2.width)
    max_height = max(image1.height, image2.height)

    transform = transforms.Compose([
        # transforms.Resize((max_height, max_width)),
        transforms.ToTensor()
    ])

    image1 = transform(image1)
    image2 = transform(image2)

    # Add a fourth dimension representing the batch number and compute the features
    features1 = model.extract_features(image1.unsqueeze(0))
    features2 = model.extract_features(image2.unsqueeze(0))

    # Flatten the features and apply cosine similarity
    cos = nn.CosineSimilarity(dim=0)
    value = round(float(cos(features1.reshape(1, -1).squeeze(), features2.reshape(1, -1).squeeze())), 4)

    return value

print(get_similarity())