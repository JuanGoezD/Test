import streamlit as st
import pandas as pd

# Crear un DataFrame de ejemplo con información de juegos
data = {
    'nombre': ['Juego 1', 'Juego 2', 'Juego 3', 'Otra Juego 1'],
    'calificacion': [0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Función para mostrar la información de un juego y permitir calificarla
def rate_game(game_name, idx):
    """Retorna la información de un juego y permite calificarlo.

    Esta función toma un dataframe de pandas y realiza una busqueda
    por la columna nombres.
    """
    game = df[df['nombre'] == game_name].iloc[0]
    st.subheader(game['nombre'])
  
    st.write("Calificación actual:")
    st.write(f"{game['calificacion']} ★")

    new_rating = st.selectbox(f"Calificar este juego{idx}:", options=[0, 1, 2, 3, 4, 5])
    st.write(f"Has calificado {game['nombre']} con {new_rating} ★")

    # Actualizar la calificación en el DataFrame
    df.loc[df['nombre'] == game_name, 'calificacion'] = new_rating

# Encabezado de la aplicación
st.title("Buscador de Juegos")

# Barra de búsqueda
game_name = st.text_input("Buscar juegos por nombre")

# Filtrar el DataFrame según la búsqueda
if game_name:
    filtered_games = df[df['nombre'].str.contains(game_name, case=False, na=False)]
    if not filtered_games.empty:
        for idx, game in filtered_games.iterrows():
            rate_game(game['nombre'], idx)
    else:
        st.write("No se encontraron juegos que coincidan con la búsqueda.")