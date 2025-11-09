import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
import pickle

#Misma idea que el Training_4 pero con mas canciones, 

df = pd.read_csv("data/Music_Features_2.csv")  
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#print(df['genre'].value_counts())

# genre
# Rock                   3097
# Electronic             3073
# Punk                   2584
# Experimental           1801
# Hip-Hop                1761
# Folk                   1215
# Chiptune / Glitch      1181
# Instrumental           1045
# Pop                     945
# International           814
# Ambient Electronic      796
# Classical               495
# Old-Time / Historic     408
# Jazz                    306
# Country                 142
# Spoken                   94
# Soul-RnB                 94
# Blues                    58
# Easy Listening           13

#Eliminamos aquellos generos que pueden ser "confusos" y aquellos que tengan pocos ejemplos

# Lista de generos a conservar
genres_to_keep = [
    'Rock', 'Electronic', 'Punk',
    'Hip-Hop', 'Folk',
    'Pop', 'Classical'
]

df = df[df['genre'].isin(genres_to_keep)].reset_index(drop=True)

#hacemos undersampling para que los generos con mas ejemplos no desbalanceen el dataset

target_size = 1700
balanced_list = []

# Recorrer cada género
for genre, group in df.groupby('genre'):
    if len(group) > target_size:
        # Reducir (sample aleatorio sin reemplazo)
        group_downsampled = group.sample(n=target_size, random_state=42)
        balanced_list.append(group_downsampled)
    else:
        # Dejar igual si tiene <= target_size
        balanced_list.append(group)

# Combinar todo y mezclar filas
df = pd.concat(balanced_list).sample(frac=1, random_state=42).reset_index(drop=True)

#Ahora hacemos oversampling para pop y classical ya que son los que menos datos tienen

target_size = 1200

df_pop = df[df['genre'] == 'Pop']
df_classical = df[df['genre'] == 'Classical']
df_otros = df[~df['genre'].isin(['Pop', 'Classical'])]

# Oversampling solo en Jazz y Electronic
df_pop_upsampled = resample(df_pop,
                             replace=True,
                             n_samples=target_size,
                             random_state=42)

df_classical_upsampled = resample(df_classical,
                                   replace=True,
                                   n_samples=target_size,
                                   random_state=42)

df_balanced = pd.concat([df_otros, df_pop_upsampled, df_classical_upsampled])
df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) #evita que los duplicados queden juntos

features = [
    'mean_stft', 'var_stft', 'tempo', 'rms_mean', 'rms_var', 'centroid_mean',
    'centroid_var', 'bandwidth_mean', 'bandwidth_var', 'rolloff_mean',
    'rolloff_var', 'crossing_mean', 'crossing_var', 'harmonic_mean',
    'harmonic_var', 'contrast_mean', 'contrast_var',
    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var',
    'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var',
    'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var',
    'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var',
    'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
    'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
    'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var',
    'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var',
    'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var',
    'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
]

X = df[features].values
y = df['genre'].values

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
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Guardamos los datos
with open("data/processed_data_4.pkl", "wb") as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoder': encoder,
        'scaler': scaler
    }, f)