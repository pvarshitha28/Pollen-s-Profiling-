import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/pollen-grain-image-classification")

print("Path to dataset files:", path)