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

#El genero Acapella solamente tiene 119 canciones, procedemos a eliminarlo

df = df[df['genre'] != 'A Capella']

#Tambien elimino anime porque no creo que sea un genero a distinguir

df = df[df['genre'] != 'Anime']

#El genero Children's music aparece dos veces con el mismo nombre, se van a eliminar los dos, para evitar juntarlos y que estos tengan
#mas canciones que los demas y quede el dataset desbalanceado. Y porque no se que genero es.

df = df[~df['genre'].isin(["Children’s Music", "Children's Music"])]

#Se agrupan géneros musicales similares para reducir la cantidad total de clases y equilibrar el dataset.

genre_mapping = {
    # Pop y derivados
    'Dance': 'Pop',

    # Rock y derivados
    'Indie': 'Rock',
    'Alternative': 'Rock',

    # Hip-Hop y Rap
    'Rap': 'Hip-Hop',

    # Soul y derivados
    'R&B': 'Soul',
    'Blues': 'Soul',

    # Reggae, Reggaeton, Ska
    'Reggaeton': 'Reggae',
    'Ska': 'Reggae',

    # Classical y Opera
    'Opera': 'Classical',

    # Folk y World
    'World': 'Folk',
    'Country': 'Folk',

    # Soundtrack y similares
    'Movie': 'Soundtrack',
    'Comedy': 'Soundtrack'
}

df['genre'] = df['genre'].replace(genre_mapping)


#print(df['genre'].value_counts())


#Los generos jazz y electronica tienen 9000 canciones mientras generos como el rock y pop tiene casi 30k
#para evitar el desbalanceo hacemos oversampling, para duplicar aleatoriamente canciones de estos generos
#Intentamos no generar muchas duplicaciones

# Definir el tamaño objetivo 
target_size = 20000

# Separar cada subconjunto
df_jazz = df[df['genre'] == 'Jazz']
df_electronic = df[df['genre'] == 'Electronic']
df_otros = df[~df['genre'].isin(['Jazz', 'Electronic'])]

# Oversampling solo en Jazz y Electronic
df_jazz_upsampled = resample(df_jazz,
                             replace=True,
                             n_samples=target_size,
                             random_state=42)

df_electronic_upsampled = resample(df_electronic,
                                   replace=True,
                                   n_samples=target_size,
                                   random_state=42)

df_balanced = pd.concat([df_otros, df_jazz_upsampled, df_electronic_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) #evita que los duplicados queden juntos

#los features key (nota musical, Do, Re, Mi, etc) y mode (si es triste o alegre) tienen valores no
#numericos, mapeamos los valores para poder operar con ellos

key_mapping = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
mode_mapping = {'Minor': 0, 'Major': 1}

df_balanced['key'] = df_balanced['key'].map(key_mapping).fillna(-1)
df_balanced['mode'] = df_balanced['mode'].map(mode_mapping).fillna(-1)


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

#features
features = [
    'danceability','energy','valence','tempo','acousticness',
    'instrumentalness','liveness','speechiness','key','mode','loudness'
]
X = df_balanced[features].values
y = df_balanced['genre'].values  # columna con el género

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
with open("data/processed_data.pkl", "wb") as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoder': encoder,
        'scaler': scaler
    }, f)