from inicio import local_css
import streamlit as st
import pandas as pd
import requests

# Configura el título y el favicon de la página
st.set_page_config(
    page_title="Gamer's Companion 🎮",
    page_icon="🎮",
)

local_css("style.css")

# Configura tu clave API de IGDB
api_key = '8h1ymcezojqdpcvmz5fvwxal2myoxp'


def get_game_info(game_name):
    """Define la URL y los encabezados para la solicitud de la API."""
    url = 'https://api.igdb.com/v4/games'
    headers = {'Client-ID': 'ju1vfy05jqstzoclqv1cs2hsomw1au', 'Authorization': f'Bearer {api_key}'}


    # Define la consulta para buscar el juego
    body = f'''
    fields name, summary, involved_companies.company.name, platforms.name, cover.url;
    where name ~ "{game_name}";'''
    
    # Realiza la solicitud a la API
    response = requests.post(url, headers=headers, data=body)
    
    # Devuelve los datos del juego
    return response.json()

# Crea una barra de búsqueda en Streamlit
game_name = st.text_input('Busca un videojuego')

# Si se introduce un nombre de juego, busca la información del juego
if game_name:
    game_info = get_game_info(game_name)
    
    # Verifica si game_info contiene algún elemento
    if game_info:
        # Muestra el nombre del juego en el centro en la parte superior
        st.write(f"<h1 style='text-align: center;'>{game_info[0]['name']}</h1>"
                 if 'name' in game_info[0] else "Nombre no disponible",
                 unsafe_allow_html=True)
        
        # Crea dos columnas para mostrar el cover y la información del juego
        col1, col2 = st.columns(2)
        
        # Muestra la portada del juego en la columna de la izquierda
        if 'cover' in game_info[0] and 'url' in game_info[0]['cover']:
            image_url = ('https://images.igdb.com/igdb/image/upload/t_cover_big/'
                         + game_info[0]['cover']['url'].split('/')[-1])
            col1.image(image_url, use_column_width=True)
        else:
            st.write("Imagen no disponible")


        # Muestra la información del juego en la columna de la derecha
        col2.write(f"**Sinopsis:** {game_info[0]['summary']}" if 'summary' in game_info[0] else "Sinopsis no disponible")
        col2.write(f"**Desarrollador:** {game_info[0]['involved_companies'][0]['company']['name']}" if 'involved_companies' in game_info[0] and game_info[0]['involved_companies'] else "Desarrollador no disponible")
        col2.write(f"**Plataformas:** {', '.join([platform['name'] for platform in game_info[0]['platforms']])}" if 'platforms' in game_info[0] and game_info[0]['platforms'] else "Plataformas no disponibles")
    else:
        st.write("Lo siento, no pude encontrar ningún juego con ese nombre.")
