import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
import pickle

# Vamos a entrenar una red neuronal de clasificación que, a partir de distintas características de una canción 
# (como energy, key, tempo, cantidad de palabras, acústica, etc.), será capaz de predecir su género musical.

# The dataset is collated from Spotify's API using two separate python scripts to extract popular '
# 'and non-popular songs and their associated audio and descriptive features. '
# 'Descriptive features of a song include information about the song such as the artist name, album name and release date. '
# 'Audio features include key, valence , danceability and energy which are results of spotify's audio analysis.

#Carga dataset
df = pd.read_csv("data/SpotifyFeatures.csv")  

selected_genres = [
    'Soundtrack', 'Jazz', 'Pop', 'Electronic',
    'Folk', 'Rock', 'Classical',
    'Rap', 'Blues', 'Anime',
    'Reggaeton', 'Country', 'Opera'
]

# Filtrar el DataFrame
df_filtered = df[df['genre'].isin(selected_genres)].copy()

# Revisar la distribución después del filtrado
print(df_filtered['genre'].value_counts())



#los features key (nota musical, Do, Re, Mi, etc) y mode (si es triste o alegre) tienen valores no
#numericos, mapeamos los valores para poder operar con ellos

key_mapping = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
mode_mapping = {'Minor': 0, 'Major': 1}

time_map = {
    '0/4': 0,
    '1/4': 1,
    '3/4': 3,
    '4/4': 4,
    '5/4': 5
}

df_filtered['time_signature'] = df_filtered['time_signature'].map(time_map)
df_filtered['key'] = df_filtered['key'].map(key_mapping).fillna(-1)
df_filtered['mode'] = df_filtered['mode'].map(mode_mapping).fillna(-1)


# danceability: Qué tan “bailable” es la canción, basado en ritmo, estabilidad y regularidad.
# energy: Nivel de energía de la canción (intensa, fuerte vs suave, tranquila).
# valence: Positividad de la canción; alto = feliz, bajo = triste o melancólica.
# tempo: Velocidad de la canción en BPM (beats por minuto).
# acousticness: Probabilidad de que la canción sea acústica; alto = más acústica.
# instrumentalness: Probabilidad de que la canción no tenga voces.
# liveness:  Indica si la canción fue grabada en vivo; alto = grabación en vivo.
# speechiness: Cantidad de palabras habladas; alto = más parecido a spoken word o rap.
# key: Tonalidad de la canción (0 a 11, cada número representa una nota: C, C#, D…).
# mode: Mayor (1) o menor (0), influye en el tono emocional (mayor = más alegre).
# loudness:  Volumen promedio de la canción en decibelios (dB).
# time_signature: Firma de compás, ej. 4/4, 3/4; indica estructura rítmica.


#agregamos time_signature', 'popularity a los features
#features
features = [
    'danceability','energy','valence','tempo','acousticness',
    'instrumentalness','liveness','speechiness','key','mode','loudness', 'time_signature', 'popularity'
]
X = df_filtered[features].values
y = df_filtered['genre'].values  # columna con el género

#Convertimos los textos de generos en el enconder y los transformamos a numeros
encoder = LabelEncoder()
y = encoder.fit_transform(y)  

# print(df['genre'].value_counts())
# print(X[:5])
# print(y[:5])
# print(encoder.classes_)

# One-hot encoding
y_encoded = to_categorical(y)

#uso StandardScaler para estandarizar 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#divido datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y
)

 # Guardamos los datos
with open("data/processed_data_2.pkl", "wb") as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoder': encoder,
        'scaler': scaler
    }, f)