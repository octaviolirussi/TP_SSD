import pickle
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ---------- CARGAR DATOS ----------
with open("data/processed_data_4.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
encoder = data['encoder']
scaler = data['scaler']


cant_features = X_train.shape[1]
cant_genre = y_train.shape[1]

# 🚀 Mucho más grande, sin apenas regularización
model = Sequential([
    Dense(1024, activation='relu', input_shape=(cant_features,)),  
    BatchNormalization(),                                          
    Dropout(0.05),                                               

    Dense(768, activation='relu'),
    BatchNormalization(),
    Dropout(0.05),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.05),

    Dense(256, activation='relu'),
    BatchNormalization(),

    Dense(cant_genre, activation='softmax')
])


optimizer = Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,            
    min_delta=1e-5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

genre_names = list(encoder.classes_) 
y_train_int = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weights = dict(enumerate(class_weights))

rock_idx = next(i for i, g in enumerate(genre_names) if 'rock' in g.lower())
punk_idx = next(i for i, g in enumerate(genre_names) if 'punk' in g.lower())

class_weights[rock_idx] *= 1.8
class_weights[punk_idx] *= 1.8

callbacks = [early_stop, reduce_lr]

# 🧠 Entrenar con muchas épocas → más riesgo de memorizar
history = model.fit(
    X_train, y_train,
    epochs=500,                
    batch_size=32,             
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)
#=============================================== Graficos ==========================================================================
# Loss
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)  # 1 fila, 2 columnas, primera gráfica
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.title('Loss durante el entrenamiento')
plt.legend()

# Accuracy
plt.subplot(1,2,2)  # 1 fila, 2 columnas, segunda gráfica
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.title('Accuracy durante el entrenamiento')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)

y_test_cat = np.argmax(y_test,axis=1)
y_pred_cat = np.argmax(y_pred,axis=1)

class_labels = encoder.classes_
y_labels = class_labels[y_pred_cat]
y_true_labels = class_labels[y_test_cat]

# Matriz de confusión
cm = confusion_matrix(y_true_labels, y_labels, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Purples, xticks_rotation=45)  
plt.title("Confusion Matrix")
plt.show()

# Reporte de clasificación
print("Classification report:\n")
print(classification_report(y_test_cat, y_pred_cat, target_names=class_labels))