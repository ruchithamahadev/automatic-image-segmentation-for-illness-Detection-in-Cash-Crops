import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
classes = 6

data = []
labels = []
for i in range(classes):
    p = base / f"dataset/train/{i}"
    if not p.exists(): continue
    for imgf in p.iterdir():
        try:
            img = Image.open(imgf).resize((30,30))
            data.append(np.array(img))
            labels.append(i)
        except:
            pass

data = np.array(data); labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, classes); y_test = to_categorical(y_test, classes)

model = Sequential([
    Conv2D(8,(3,3),activation='relu', input_shape=X_train.shape[1:]),
    MaxPool2D((2,2)),
    Conv2D(16,(3,3),activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test), batch_size=8)
model.save(str(base / 'model' / 'my_model.h5'))
# plot training
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.show()
