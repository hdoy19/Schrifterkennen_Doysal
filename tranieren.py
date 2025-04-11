import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
from sklearn.model_selection import train_test_split

#Daten laden
images = np.load("images.npy")           
labels = np.load("labels.npy")          

#Normalisieren und Labels vorbereiten
labels = labels.astype('int32').flatten()  

#Trainings-/Testdaten aufteilen (90% / 10%)
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42)

#Feedforward-Netzwerk definieren
model = Sequential([
    Flatten(input_shape=(28, 28)),     # 784 Inputs
    Dense(128, activation='relu'),    # Hidden Layer 1
    Dense(64, activation='relu'),     # Hidden Layer 2
    Dense(26, activation='softmax')   # 26 Klassen (A–Z)
])

#Kompilieren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Trainieren
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

#Testen
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n Test Accuracy: {test_acc * 100:.2f}%")

#Modell speichern
model.save("mein_tf_model.h5")
print("Modell gespeichert als 'mein_tf_model.h5'")
