from src.train import train_model
from src.config import Config

def main():
    
    # Get the number of epochs from the user
    try:
        epochs = int(input("Enter the number of epochs for training: "))
        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than zero.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    # Train the model
    print("Training the model...")
    try:
        train_model(epochs=epochs)
        print("Model trained successfully! The model has been saved.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()
