from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from src.model import create_cnn_model
from src.preprocess import preprocess_data
from src.config import Config
import os
from datetime import datetime

def train_model(epochs=Config.EPOCHS):
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Build the model
    model = create_cnn_model(Config.IMG_SIZE)

    # Create log file
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Open the log file
    log_file = open(log_file_name, "w")

    # Custom callback to log training progress
    class FileLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_message = (
                f"Epoch {epoch+1}/{epochs}:\n"
                f"  - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}\n"
                f"  - Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}\n"
            )
            print(log_message.strip())  # Print to console
            log_file.write(log_message)  # Write to file

    # Train the model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=Config.BATCH_SIZE,
        validation_split=Config.VALIDATION_SPLIT,
        callbacks=[FileLogger()]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    final_log = f"Test Accuracy: {test_acc:.2f}\n"
    print(final_log.strip())
    log_file.write(final_log)

    # Save the model
    model.save("face_recognition_model.h5")
    final_save_log = "Model saved as face_recognition_model.h5\n"
    print(final_save_log.strip())
    log_file.write(final_save_log)

    # Close the log file
    log_file.close()

    print(f"Training logs saved to: {log_file_name}")
    return model, X_test, y_test
