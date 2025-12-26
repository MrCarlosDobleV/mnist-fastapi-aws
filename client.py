import requests
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

API_URL = "http://3.129.92.18:8000/predict" # Replace with the IP of your deployed EC2 instance

def main():
    print("[1] Starting client")

    # Same transform used in training
    print("[2] Setting up transforms")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST test set (NO re-download)
    print("[3] Loading MNIST test dataset from ./data")
    test_dataset = datasets.MNIST(
        "./data", train=False, download=False, transform=transform
    )

    # Pick ONE sample
    index = 5
    print(f"[4] Loading sample index {index}")
    image, true_label = test_dataset[index]

    # Show image
    print("[5] Displaying image")
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"True label: {true_label}")
    plt.axis("off")
    plt.show()

    # Convert tensor to JSON-serializable list
    print("[6] Preparing image for API request")
    image_list = image.squeeze().tolist()

    # Send to API
    print("[7] Sending request to API")
    response = requests.post(
        API_URL,
        json={"image": image_list}
    )

    print("[8] Receiving response from API")
    result = response.json()

    print("[9] Printing results")
    print("True label :", true_label)
    print("Prediction :", result["prediction"])
    print("Confidence :", f"{result['confidence']:.4f}")

    print("[10] Client finished successfully")

if __name__ == "__main__":
    main()
