import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

# Modell laden
model = tf.keras.models.load_model("dateiname.keras")  # Passe den Pfad an

# Mappings für die Buchstaben (anpassen, je nachdem wie dein Modell trainiert ist)
klassen = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Fenster
root = tk.Tk()
root.title("Buchstabenerkennung")

# Canvas zum Zeichnen
canvas_width, canvas_height = 200, 200
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Bild zum Zeichnen (für spätere Konvertierung)
image1 = Image.new("L", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(image1)

# Zeichnen per Maus
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill="black")

canvas.bind("<B1-Motion>", paint)

# Bild verarbeiten und vorhersagen
def predict():
    # Bild vorbereiten
    img = image1.copy()
    img = ImageOps.invert(img)
    img = img.resize((28, 28))  # Muss zur Eingabegröße deines Modells passen
    img_np = np.array(img) / 255.0
    img_np = img_np.reshape(1, 28, 28, 1)  # Falls CNN mit 1 Channel

    # Vorhersage
    prediction = model.predict(img_np)[0]
    index = np.argmax(prediction)
    ergebnis_label.config(text=f"Vorhersage: {klassen[index]}")

    # Optional: Visualisierung der Wahrscheinlichkeiten
    plt.clf()
    plt.bar(klassen, prediction)
    plt.title("Vorhersage-Wahrscheinlichkeiten")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pred_img = Image.open(buf)
    pred_img.show()

# Buttons
btn_predict = tk.Button(root, text="Erkennen", command=predict)
btn_predict.pack()

btn_clear = tk.Button(root, text="Löschen", command=lambda: [canvas.delete("all"), draw.rectangle([0,0,canvas_width,canvas_height], fill="white")])
btn_clear.pack()

# Label für Ergebnis
ergebnis_label = tk.Label(root, text="Vorhersage: ")
ergebnis_label.pack()

root.mainloop()
