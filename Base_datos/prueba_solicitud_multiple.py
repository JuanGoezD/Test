import requests
import json

# URL de la API de IGDB

url = "https://api.igdb.com/v4/games"

# Headers de la solicitud

headers = {
    "Client-ID": "ju1vfy05jqstzoclqv1cs2hsomw1au",
    "Authorization": "Bearer 8h1ymcezojqdpcvmz5fvwxal2myoxp",
    
}

# Parámetros de la solicitud
for i in [1942,1943,1944]:
    body = f'fields name,rating,cover.url; where id = {i};'

# Realizar la solicitud

    response = requests.post(url, headers=headers, data=body)

# Procesar la respuesta

    if response.status_code == 200:
        juegos = response.json()
    else:
        raise Exception(f"Error al obtener los videojuegos: {response.status_code}")

# Imprimir los videojuegos
    
    print(juegos)