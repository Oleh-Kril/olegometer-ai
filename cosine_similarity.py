from efficientnet_pytorch import EfficientNet

from torch import nn

model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()

# Load the model
def get_similarity(
    image1,
    image2
):
    # Add a fourth dimension representing the batch number and compute the features
    features1 = model.extract_features(image1.unsqueeze(0))
    features2 = model.extract_features(image2.unsqueeze(0))

    # Flatten the features and apply cosine similarity
    cos = nn.CosineSimilarity(dim=0)
    value = round(float(cos(features1.reshape(1, -1).squeeze(), features2.reshape(1, -1).squeeze())), 4)

    return value
