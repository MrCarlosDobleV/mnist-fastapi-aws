# MNIST AWS Classifier

End-to-end MNIST digit classification project designed for deployment on **AWS Free Tier**.

The project follows a simple and production-oriented workflow:
- Train a neural network **locally**
- Save the trained model
- Serve predictions through a **FastAPI** inference service
- Deployable on AWS EC2 (t2.micro)

---

## Project Structure

```text
mnist-aws-classifier/
├── train.py          # Model training script
├── api.py            # FastAPI inference service
├── mnist_model.pt    # Trained model (generated locally)
├── requirements.txt  # Package requirements
├── .dockerignore
├── .gitignore
└── README.md
