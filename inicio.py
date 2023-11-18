# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Function to create or retrieve DataFrame from session state
def get_points() -> None:
    """
    Retrieve or create a DataFrame for storing points.

    Args:
        None
    
    Returns:
        st.session_state.points: The points added to the DataFrame
    """
    if 'points' not in st.session_state:
        st.session_state.points = pd.DataFrame(columns=['Latitude', 'Longitude', 'Label'])
    return st.session_state.points

# Initialize or get the DataFrame for storing points
points_df = get_points()

# Input fields for latitude, longitude, and label
new_lat = st.number_input('Enter Latitude:')
new_lon = st.number_input('Enter Longitude:')
new_label = st.text_input('Enter Label (Optional)', value='')

# Add button to append points to the DataFrame
if st.button('Add Point'):
    new_point = pd.DataFrame({'Latitude': [new_lat], 'Longitude': [new_lon], 'Label': [new_label]})
    points_df = pd.concat([points_df, new_point], ignore_index=True)
    st.session_state.points = points_df  # Store updated points in session state
    st.experimental_rerun()  # Refresh the page to clear the input fields

# Display points added by the user
st.write("### Points Added by User")
st.write(points_df)

# Button to display all possible connections
display_connections = st.button('Display Connections')

# Check if the button is clicked and points exist in the DataFrame
if display_connections:
    if not points_df.empty:
        # Plot all possible connections between points
        plt.figure(figsize=(8, 6))
        
        # Iterate through each pair of points to plot connections
        for i, point1 in points_df.iterrows():
            for j, point2 in points_df.iterrows():
                if i != j:  # Avoid plotting connections for the same point
                    # Plot a dashed line between two points
                    plt.plot([point1['Longitude'], point2['Longitude']], 
                             [point1['Latitude'], point2['Latitude']], 'k--')
        
        # Set labels and title for the plot
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('All Possible Connections between Points')
        
        # Display the plot using Streamlit
        st.pyplot(plt)  
    else:
        st.write("No points to display.")  # Inform if there are no points to connect

# Select starting point for solving the TSP
starting_point = st.selectbox('Select Starting Point', options=list(points_df['Label']))

# Button to solve the TSP
solve_tsp = st.button('Solve the problem')
if solve_tsp:
    if not points_df.empty:
        # Get the number of points in the dataset
        num_points = len(points_df)
        
        # Create an empty distance matrix to store distances between points
        dist_matrix = np.zeros((num_points, num_points))
        
        # Calculate the distances between each pair of points
        for i in range(num_points):
            for j in range(num_points):
                # Calculate the differences in latitude and longitude
                lat_diff = points_df.iloc[i]['Latitude'] - points_df.iloc[j]['Latitude']
                lon_diff = points_df.iloc[i]['Longitude'] - points_df.iloc[j]['Longitude']
                
                # Calculate the Euclidean distance between two points
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                
                # Store the calculated distance in the distance matrix
                dist_matrix[i][j] = distance

        # Use linear_sum_assignment to solve the Traveling Salesman Problem
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # Initialize an empty list to store the labels in the optimal order
        optimal_order = []

        # Retrieve the labels (cities) in the order of the optimal path
        for i in col_ind:
            label = points_df.iloc[i]['Label']
            optimal_order.append(label)

        # Rearrange the order based on the selected starting point
        starting_index = optimal_order.index(starting_point)
        optimal_order = optimal_order[starting_index:] + optimal_order[:starting_index]

        # Plot the route taken in the TSP
        plt.figure(figsize=(8, 6))

        # Plotting the optimal route connecting the cities
        for i in range(num_points - 1):
            # Retrieve the start and end points for each segment of the route
            start_point = points_df[points_df['Label'] == optimal_order[i]].iloc[0]
            end_point = points_df[points_df['Label'] == optimal_order[i+1]].iloc[0]
            
            # Plot a line between the start and end points
            plt.plot([start_point['Longitude'], end_point['Longitude']], 
                    [start_point['Latitude'], end_point['Latitude']], 'g--')

        # Connect the last point to the first one to complete the route
        start_point = points_df[points_df['Label'] == optimal_order[-1]].iloc[0]
        end_point = points_df[points_df['Label'] == optimal_order[0]].iloc[0]
        plt.plot([start_point['Longitude'], end_point['Longitude']], 
                [start_point['Latitude'], end_point['Latitude']], 'g--')

        # Labels to the graph
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Optimal Route for TSP')

        # Display the plot using Streamlit
        st.pyplot(plt)  

        # Display the order of visiting cities
        st.write("Optimal Order of Visiting Cities:")
        # Display the cities
        st.write(" -> ".join(optimal_order))  
    else:
        st.write("No points to display.")
