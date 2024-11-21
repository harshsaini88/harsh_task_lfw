import random
import numpy as np

def alert_random_image(model, X_test, y_test, log_func=None):
    random_idx = random.randint(0, X_test.shape[0] - 1)
    random_image = X_test[random_idx]
    random_label = np.argmax(y_test[random_idx])

    # Predict category
    predicted_category = model.predict(np.expand_dims(random_image, axis=0))
    predicted_label = np.argmax(predicted_category)

    if log_func:
        log_func(f"Actual Label: {'Category A' if random_label == 0 else 'Category B'}")
        log_func(f"Predicted Label: {'Category A' if predicted_label == 0 else 'Category B'}")
    
    if predicted_label == 0:
        if log_func:
            log_func("ALERT: The selected image belongs to Category A!")
    else:
        if log_func:
            log_func("The selected image does not belong to Category A.")
