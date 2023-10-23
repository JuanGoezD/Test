import pandas as pd

# Cargamos la el dataset através de una URl
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ7GUrAS\
-ws9zp3F8v8KGnQmnb6K5Aw6PEcLlueatCXMakxJxkB9FmA0z1kIAxbte3cn\
TrbiDtexQ4g/pub?gid=207569464&single=true&output=csv"

# Cargamos el dataset
dataset = pd.read_csv(url)

# filtramos los juegos del dataset por rating

dataset = dataset.sort_values(by="rating", ascending = False)

# Mostramos el top 50
print(f'{dataset[['name','rating']].head(50)} \n')

# El dataset tiene filas repetidas para el mismo juego
# por lo que eliminaremos las repetidas

dataset = dataset.drop_duplicates("name", keep="first")

# Mostramos el top50 sin filas repetidas
print(dataset[['name','rating']].head(50))

# Descargamos el dataset limpio y ordenado
dataset.to_csv('dataset_videojuegos')