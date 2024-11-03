import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def get_discrete_annotations_variance():
    # Consulta de todos los escaneos en el dataset
    scans = pl.query(pl.Scan)
    
    # Inicialización de listas para almacenar la varianza de cada propiedad
    sphericities_var = []
    textures_var = []
    malignancies_var = []
    calcifications_var = []
    internal_structures_var = []
    margins_var = []
    spiculations_var = []
    subtleties_var = []
    lobulations_var = []
    
    # Iteración sobre todos los escaneos del dataset
    for scan in scans:
        # Iteración sobre todos los nódulos en cada escaneo
        for nodule in scan.cluster_annotations():
            # Inicialización de listas para acumular las propiedades de cada anotación en el nódulo
            sphericities = []
            textures = []
            malignancies = []
            calcifications = []
            internal_structures = []
            margins = []
            spiculations = []
            subtleties = []
            lobulations = []
            
            # Recolección de las propiedades de cada anotación
            for annotation in nodule:
                sphericities.append(annotation.sphericity)
                textures.append(annotation.texture)
                malignancies.append(annotation.malignancy)
                calcifications.append(annotation.calcification)
                internal_structures.append(annotation.internalStructure)
                margins.append(annotation.margin)
                spiculations.append(annotation.spiculation)
                subtleties.append(annotation.subtlety)
                lobulations.append(annotation.lobulation)
            
            # Cálculo de la varianza para cada propiedad y almacenamiento en las listas correspondientes
            sphericities_var.append(np.var(sphericities))
            textures_var.append(np.var(textures))
            malignancies_var.append(np.var(malignancies))
            calcifications_var.append(np.var(calcifications))
            internal_structures_var.append(np.var(internal_structures))
            margins_var.append(np.var(margins))
            spiculations_var.append(np.var(spiculations))
            subtleties_var.append(np.var(subtleties))
            lobulations_var.append(np.var(lobulations))
   
    # Retorno de las varianzas de todas las propiedades
    #return (sphericities_var, textures_var, malignancies_var, calcifications_var, 
    #        internal_structures_var, margins_var, spiculations_var, subtleties_var, lobulations_var)


def plot_all_patients_annotations(plot=False):
    # Initialize lists to collect annotation properties across all patients
    sphericities = []
    volumes = []
    surface_areas = []
    textures = []
    malignancies = []
    calcifications = []
    internal_structures = []
    margins = []
    spiculations = []
    subtleties = []
    diameters = []
    lobulations = []

    # Query all scans in the dataset
    scans = pl.query(pl.Scan)

    # Iterate over all scans in the dataset
    for scan in scans:
        # Iterate over all nodules in the scan
        for nodule in scan.cluster_annotations():
            for annotation in nodule:
                # Collect properties of each annotation
                sphericities.append(annotation.sphericity)
                volumes.append(annotation.volume)
                surface_areas.append(annotation.surface_area)
                textures.append(annotation.texture)
                malignancies.append(annotation.malignancy)
                calcifications.append(annotation.calcification)  
                internal_structures.append(annotation.internalStructure)  # Usar internal_structure (en lugar de internalStructure)
                margins.append(annotation.margin)  
                spiculations.append(annotation.spiculation)  
                subtleties.append(annotation.subtlety) 
                diameters.append(annotation.diameter) 
                lobulations.append(annotation.lobulation)  
    
    # If the 'plot' flag is True, generate the plots
    if plot:

        # Create subplots for the histograms
        fig, axs = plt.subplots(6, 2, figsize=(12, 24))

        # Bar plot for sphericity
        sphericity_bins = np.linspace(0, 1, 11)  # Create 10 bins from 0 to 1 for sphericity
        axs[0, 0].bar(np.unique(sphericities), [sphericities.count(x) for x in np.unique(sphericities)], color='skyblue')
        axs[0, 0].set_title('Sphericity')
        axs[0, 0].set_xlabel('Sphericity Value')
        axs[0, 0].set_ylabel('Frequency')

        # Bar plot for volume
        volume_bins = np.linspace(0, max(volumes), 20)  # Create 20 bins for volume
        volume_counts, _ = np.histogram(volumes, bins=volume_bins)
        axs[0, 1].bar(volume_bins[:-1], volume_counts, width=np.diff(volume_bins), color='lightgreen', align='edge')
        axs[0, 1].set_title('Volume')
        axs[0, 1].set_xlabel('Volume (cubic mm)')
        axs[0, 1].set_ylabel('Frequency')

        # Bar plot for surface area
        surface_area_bins = np.linspace(0, max(surface_areas), 20)  # Create 20 bins for surface area
        surface_area_counts, _ = np.histogram(surface_areas, bins=surface_area_bins)
        axs[1, 0].bar(surface_area_bins[:-1], surface_area_counts, width=np.diff(surface_area_bins), color='lightcoral', align='edge')
        axs[1, 0].set_title('Surface Area')
        axs[1, 0].set_xlabel('Surface Area (square mm)')
        axs[1, 0].set_ylabel('Frequency')

        # Bar plot for diameter
        diameter_bins = np.linspace(0, max(diameters), 20)  # Create 20 bins for diameter
        diameter_counts, _ = np.histogram(diameters, bins=diameter_bins)
        axs[1, 1].bar(diameter_bins[:-1], diameter_counts, width=np.diff(diameter_bins), color='lightblue', align='edge')
        axs[1, 1].set_title('Diameter')
        axs[1, 1].set_xlabel('Diameter (mm)')
        axs[1, 1].set_ylabel('Frequency')

        # Bar plot for texture
        axs[2, 0].bar(np.unique(textures), [textures.count(x) for x in np.unique(textures)], color='lightpink')
        axs[2, 0].set_title('Texture (1-5)')
        axs[2, 0].set_xlabel('Texture Rating')
        axs[2, 0].set_ylabel('Frequency')

        # Bar plot for malignancy
        axs[2, 1].bar(np.unique(malignancies), [malignancies.count(x) for x in np.unique(malignancies)], color='lightgray')
        axs[2, 1].set_title('Malignancy (1-5)')
        axs[2, 1].set_xlabel('Malignancy Rating')
        axs[2, 1].set_ylabel('Frequency')

        # Bar plot for calcification
        axs[3, 0].bar(np.unique(calcifications), [calcifications.count(x) for x in np.unique(calcifications)], color='orange')
        axs[3, 0].set_title('Calcification')
        axs[3, 0].set_xlabel('Calcification Type')
        axs[3, 0].set_ylabel('Frequency')

        # Bar plot for internal structure
        axs[3, 1].bar(np.unique(internal_structures), [internal_structures.count(x) for x in np.unique(internal_structures)], color='purple')
        axs[3, 1].set_title('Internal Structure')
        axs[3, 1].set_xlabel('Structure Type')
        axs[3, 1].set_ylabel('Frequency')
        
        # Bar plot for margins
        axs[4, 0].bar(np.unique(margins), [margins.count(x) for x in np.unique(margins)], color='yellow')
        axs[4, 0].set_title('Margins')
        axs[4, 0].set_xlabel('Margin Type')
        axs[4, 0].set_ylabel('Frequency')

        # Bar plot for spiculations
        axs[4, 1].bar(np.unique(spiculations), [spiculations.count(x) for x in np.unique(spiculations)], color='red')
        axs[4, 1].set_title('Spiculations')
        axs[4, 1].set_xlabel('Spiculation Level')
        axs[4, 1].set_ylabel('Frequency')

        # Bar plot for subtleties
        axs[5, 0].bar(np.unique(subtleties), [subtleties.count(x) for x in np.unique(subtleties)], color='green')
        axs[5, 0].set_title('Subtlety')
        axs[5, 0].set_xlabel('Subtlety Rating')
        axs[5, 0].set_ylabel('Frequency')

        # Bar plot for lobulations
        axs[5, 1].bar(np.unique(lobulations), [lobulations.count(x) for x in np.unique(lobulations)], color='blue')
        axs[5, 1].set_title('Lobulation')
        axs[5, 1].set_xlabel('Lobulation Level')
        axs[5, 1].set_ylabel('Frequency')

        # Adjust layout and hide unused subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle("Annotation Properties Plots for All Patients", fontsize=16)
        plt.show()
        
        # Adjust layout and hide unused subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle("Annotation Properties Plots for All Patients", fontsize=16)
        plt.show()
    
    # Return all the collected properties
    return (sphericities, volumes, surface_areas, textures, malignancies, calcifications, 
            internal_structures, margins, spiculations, subtleties, diameters, lobulations)


def plot_CT(data,cmap = 'gray') : 
    
    if data.ndim != 2:  # Si el arreglo no es 2D
        raise ValueError("Array must be 2d")
    
    plt.imshow(data, cmap = cmap)  # Muestra la imagen en escala de grises
    plt.axis('off')  # Oculta los ejes
    plt.show()
    

def pre_post_HU(data_pre, data_post):
    plt.figure(figsize=(12, 6))
        
    # Plot before HU processing
    plt.subplot(1, 2, 1)
    plt.imshow(data_pre, cmap='gray')
    plt.title('Before HU Processing')
    plt.colorbar()

    # Plot after HU processing
    plt.subplot(1, 2, 2)
    plt.imshow(data_post, cmap='gray')
    plt.title('After HU Processing')
    plt.colorbar()

    plt.suptitle("CT Scan: Before and After HU Processing", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_malignancies(malignancies_array) : 
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create the bar plot for the different malignancy categories (1-5)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=malignancies_array, palette='Blues')

    # Titles and labels
    plt.title('Distribution of Nodule Malignancy', fontsize=16)
    plt.xlabel('Malignancy Level', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Display the plot
    plt.show()
    

def plot_malignancy_array_lenght(df) :
    # Calcular las longitudes de las listas en la columna 'annotations'
    lengths = df['annotations'].apply(len)

    # Obtener las frecuencias de cada longitud
    frequency = lengths.value_counts().sort_index()

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    bars = plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')

    # Añadir etiquetas sobre las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=12)

    plt.xticks(range(1, 9))  # Configurar los ticks del eje X del 1 al 8
    plt.xlabel('Length of Annotations')
    plt.ylabel('Frequency')
    plt.title('Frequency of Annotation Lengths for Each Nodule', fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añadir una cuadrícula horizontal
    plt.tight_layout()  # Ajustar el diseño para que se vea bien

    plt.show()
    


def plot_co_ocurrency_matrix(data) : 
    lista_anotaciones = [item[2] for item in data]
    valores = [1, 2, 3, 4, 5]

    cooc_matrix = pd.DataFrame(0, index=valores, columns=valores)

    # Llenar la matriz de co-ocurrencia
    for annotations in lista_anotaciones:
        unique_annotations = set(annotations)  # Usar un set para evitar contar duplicados
        for a in unique_annotations:
            for b in unique_annotations:
                cooc_matrix.loc[a, b] += 1

    # Mostrar la matriz de co-ocurrencia
    print("Matriz de Co-ocurrencia:")
    print(cooc_matrix)

    # Configuraciones del gráfico
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.5)  # Aumentar el tamaño de la fuente

    # Crear el heatmap
    heatmap = sns.heatmap(
        cooc_matrix,
        annot=True,
        fmt='d',
        cmap='YlGnBu',  # Paleta de colores más atractiva
        cbar_kws={'label': 'Frecuency'},  # Etiqueta para la barra de color
        linewidths=.5,  # Añadir líneas entre las celdas
        linecolor='black',  # Color de las líneas
        square=True,  # Hacer que las celdas sean cuadradas
    )


    # Títulos y etiquetas
    heatmap.set_title('Co-ocurrency Annotations Matrix', fontsize=20, fontweight='bold')
    heatmap.set_xlabel('Annotation Value', fontsize=16)
    heatmap.set_ylabel('Annotation Value', fontsize=16)

    plt.xticks(rotation=0)  # Rotar las etiquetas del eje X
    plt.yticks(rotation=0)  # Rotar las etiquetas del eje Y

    # Mostrar el gráfico
    plt.tight_layout()  # Ajustar el diseño para evitar recortes
    plt.show()