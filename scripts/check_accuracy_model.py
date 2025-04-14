import geopandas
from shapely.geometry import Point
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Confusion matrix to visualize the labels and predictions that are correct
def plot_confusion_matrix(labels,predictions,label_map,filename):
    cm = confusion_matrix(labels,predictions)
    # Visualize the confusion matrix using seaborn
    plt.figure(figsize=((len(label_map) + 2), (len(label_map)+ 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
    plt.xlabel('Predicted Continent')
    plt.ylabel('True Continent')
    plt.xticks(rotation=60)
    plt.title('Confusion Matrix for Continent Predictions')
    plt.tight_layout()
    plt.savefig(filename) # Save the plot as an image
    plt.show()

# Plot the training losses
def plot_losses(train_losses, filename):
    epochs = range(1, len(next(iter(train_losses.values()))) + 1)  # Get number of epochs from the length of the first list

    plt.figure(figsize=(10, 6))
    for loss_name, losses in train_losses.items():
        plt.plot(epochs, losses, label=f'{loss_name.capitalize()} Loss')

    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as an image
    plt.show()

# Plot the points on the world map for visualization
def plot_points_on_world_map(true_lat, true_long, predicted_lat, predicted_long, filename):
    """Plots true and predicted latitude and longitude on a world map."""
    world = geopandas.read_file("/home/chandru/binp37/data/geopandas/ne_110m_admin_0_countries.shp")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    world.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot true locations
    geometry_true = [Point(xy) for xy in zip(true_long, true_lat)]
    geo_df_true = geopandas.GeoDataFrame(geometry_true, crs=world.crs, geometry=geometry_true)  # Specify geometry
    geo_df_true.plot(ax=ax, marker='o', color='blue', markersize=15, label='True Locations')

    # Plot predicted locations
    geometry_predicted = [Point(xy) for xy in zip(predicted_long, predicted_lat)]
    geo_df_predicted = geopandas.GeoDataFrame(geometry_predicted, crs=world.crs, geometry=geometry_predicted)  # Specify geometry
    geo_df_predicted.plot(ax=ax, marker='x', color='red', markersize=15, label='Predicted Locations')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('True vs. Predicted Locations on World Map')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename) # Save the plot as an image
    plt.show()