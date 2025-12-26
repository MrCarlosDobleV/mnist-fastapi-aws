# MNIST AWS Classifier

End-to-end MNIST digit classification project designed for deployment on **AWS Free Tier**.

The project follows a simple and production-oriented workflow:
- Train a neural network **locally**
- Save the trained model
- Serve predictions through a **FastAPI** inference service
- Deployable on AWS EC2 (t2.micro)

---

## Demo video:
A short video demonstrating how a Python client sends an MNIST image to the deployed FastAPI endpoint on AWS EC2 and receives the predicted digit along with the confidence score.

[Mnist-API.webm](https://github.com/user-attachments/assets/548768cc-c0f6-4ae0-90c4-5c0ddb63806d)


## Project Structure

```text
mnist-aws-classifier/
├── train.py          # Model training script
├── api.py            # FastAPI inference service
├── mnist_model.pt    # Trained model (generated locally)
├── requirements.txt  # Package requirements
├── Dockerfile        # Dockerfile for building the project
├── .dockerignore
├── .gitignore
└── README.md


