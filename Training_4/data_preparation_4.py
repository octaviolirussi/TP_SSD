import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
import pickle

#link: https://www.kaggle.com/datasets/insiyeah/musicfeatures?select=data.csv

#A music genre is a conventional category that identifies pieces of music as belonging to a shared tradition or set of conventions. 
# It is to be distinguished from musical form and musical style. 
# The features extracted from these waves can help the machine distinguish between them.

#la idea es utilizar la informacion de los features que fueron obtenidas a partir de los archivos de audio de las canciones.
#Utilizamos las propiedades reales de las canciones, en vez de etiquetas o metadatos.

# Index(['filename', 'tempo', 'beats', 'chroma_stft', 'rmse',
#        'spectral_centroid', 'spectral_bandwidth', 'rolloff',
#        'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
#        'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
#        'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
#        'mfcc20', 'label'],
#       dtype='object')
# 30

#tempo: La velocidad a la que se reproduce un pasaje musical
#beats: Unidad rítmica en la música
#chroma_stft: promedio del espectro de cromas (notas musicales)
#rmse: energía promedio del sonido
#rmseL energía promedio del sonido
#spectral_centroid: centroide espectral (brillo del sonido)
#spectral_bandwidth: rango de frecuencias dominantes
#rolloff: frecuencia donde cae el 85% de la energía
#zero_crossing_rate: tasa de cruces por cero (ruido o percusividad)
#mfcc1–mfcc20: son valores numéricos que representan cómo suena el timbre de un audio.

#Carga dataset
df = pd.read_csv("data/Music_Features.csv")  

#todos los generos estan agrupados, los barajamos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

features = [
    'tempo', 'beats', 'chroma_stft', 'rmse',
    'spectral_centroid', 'spectral_bandwidth', 'rolloff',
    'zero_crossing_rate',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
    'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
    'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
    'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
]

X = df[features].values      # variables de entrada (features del audio)
y = df['label'].values       # columna con el género

#Convertimos los textos de generos en el enconder y los transformamos a numeros
encoder = LabelEncoder()
y = encoder.fit_transform(y)  

# One-hot encoding
y_encoded = to_categorical(y)

#uso StandardScaler para estandarizar 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#divido datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y
)

 # Guardamos los datos
with open("data/processed_data_3.pkl", "wb") as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoder': encoder,
        'scaler': scaler
    }, f)