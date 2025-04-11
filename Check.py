import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Modell und Daten laden
model = tf.keras.models.load_model("mein_tf_model.h5")
x_test = np.load("images.npy")
y_test = np.load("labels.npy").flatten()

# Zufällig 10% Testdaten
from sklearn.model_selection import train_test_split 
_, x_test, _, y_test = train_test_split(x_test, y_test, test_size=0.1, random_state=42)

# Vorhersagen berechnen
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

#umwandeln von index zu buchstabe
def label_to_char(index):
    return chr(65 + index) 

# 10 zufällige Vorhersagen anzeigen
for i in range(10):
    idx = random.randint(0, len(x_test) - 1)
    image = x_test[idx]
    true_label = y_test[idx]
    pred_label = predicted_labels[idx]

    plt.imshow(image, cmap='gray')
    plt.title(f"Vorhersagen: {label_to_char(pred_label)} | Richtiges: {label_to_char(true_label)}")
    plt.axis('off')
    plt.show()

model.save("dateiname.keras")
