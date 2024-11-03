import argparse
import os
import numpy as np
import pandas as pd
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pydicom
from statistics import  median_high

from Mask import extract_radiomics

import json

def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def calculate_malignancy(nodule):
    # Calcula la malignidad de un nódulo con las anotaciones hechas por 4 doctores.
    # Devuelve la mediana alta de la malignidad anotada y un label True o False para el cáncer.
    # Si la mediana alta es mayor a 3, devolvemos True (cáncer).
    # Si es menor a 3, devolvemos False (no cáncer).
    # Si es 3, devolvemos 'Ambiguous', para procesamiento semisupervisado futuro.
    list_of_malignancy = []
    for annotation in nodule:
        list_of_malignancy.append(annotation.malignancy)

    malignancy = median_high(list_of_malignancy)
    if malignancy > 3:
        return malignancy, True
    elif malignancy < 3:
        return malignancy, False
    else:
        return malignancy, 'Ambiguous'

# Limit the HU (Hounsfield Unit) values to a common range
def clip_hu_range(hu_image, min_hu=-1000, max_hu=400):
    """
    Clip the Hounsfield Unit (HU) values of a CT image to a specified range.
    
    Args:
        hu_image (numpy array): The input CT image in HU.
        min_hu (int, optional): The minimum HU value to clip to. Default is -1000 (typical value for air).
        max_hu (int, optional): The maximum HU value to clip to. Default is 400 (typical value for bones).
    
    Returns:
        numpy array: The CT image with HU values clipped to the range [min_hu, max_hu].
    """
    # Clip the image values to the specified range
    hu_image = np.clip(hu_image, min_hu, max_hu)
    return hu_image


# Convert DICOM image to Hounsfield Units (HU)
def convert_to_HU(archive):
    """
    Convert a DICOM image to Hounsfield Units (HU).
    
    Args:
        archive (Path or str): Path to the DICOM file.
        
    Returns:
        numpy array: The CT image converted to Hounsfield Units (HU).
    """
    # Read the DICOM file
    dicom_file = pydicom.dcmread(archive.resolve())
    # Extract the pixel data from the DICOM file
    image_array = dicom_file.pixel_array

    # Extract rescaling values from DICOM metadata (if present)
    intercept = dicom_file.RescaleIntercept if 'RescaleIntercept' in dicom_file else 0
    slope = dicom_file.RescaleSlope if 'RescaleSlope' in dicom_file else 1

    # Apply the conversion formula to transform to Hounsfield Units (HU)
    # HU = PixelValue * RescaleSlope + RescaleIntercept
    hu_image = image_array * slope + intercept

    return hu_image


# Normalize HU values between 0 and 1
def normalize_hu(hu_image, min_hu=-1000, max_hu=400):
    """
    Normalize the Hounsfield Unit (HU) values of a CT image to the range [0, 1].
    
    Args:
        hu_image (numpy array): The input CT image in HU.
        min_hu (int, optional): The minimum HU value for normalization. Default is -1000.
        max_hu (int, optional): The maximum HU value for normalization. Default is 400.
        
    Returns:
        numpy array: The CT image with HU values normalized between 0 and 1.
    """
    # Normalize the HU values to the range [0, 1]
    hu_image = (hu_image - min_hu) / (max_hu - min_hu)
    
    # Ensure that all values remain between 0 and 1
    hu_image[hu_image > 1] = 1  # Cap values greater than 1 to 1
    hu_image[hu_image < 0] = 0  # Cap values less than 0 to 0
    
    return hu_image



def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask*img

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Función para convertir arrays numpy a listas
def convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convierte arrays numpy a listas
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}  # Recursión para diccionarios
    else:
        return data

def load_json(json_path):
    """
    Carga un archivo JSON y devuelve los datos como un diccionario de Python.

    Parameters:
    - json_path (str): Ruta al archivo JSON.

    Returns:
    - dict: Contenido del archivo JSON.
    """
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def dump_json(output_json_path) : 
    # Volcar el diccionario data_storer a un archivo JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(serializable_data, json_file, indent=4)  # indent=4 para hacerlo legible
        
        

def get_features_list(data_storer):
    features_list = set()  # Utilizamos un set para evitar duplicados
    
    # Recorre cada paciente y nódulo en el JSON
    for paciente, nodulos in data_storer.items():
        
        # Recorre cada nódulo del paciente
        for nodulo, cts in nodulos.items():
            
            # Recorre cada CT dentro del nódulo
            for ct, caracteristicas in cts.items():  # for slice, diccionario de características
                for key, value in caracteristicas.items():
                    if key.startswith("original") and isinstance(value, (int, float)):
                        features_list.add(key)  # Añadir la característica al set

    return sorted(list(features_list))  # Convertir el set a una lista ordenada


def search_malignancy(df_tosearchin, patient_id, nodule, patient_col="patient_id", nodule_col="nodule_index", malignancy_col="Malignancy_al"):
    # Use boolean indexing to filter the DataFrame for the specific patient and nodule
    filtered_df = df_tosearchin[(df_tosearchin[patient_col] == patient_id) & (df_tosearchin[nodule_col] == nodule)]
    
    # Return the malignancy value or an appropriate response if no match is found
    if not filtered_df.empty:
        return filtered_df[malignancy_col].values
    else:
        return None  # Or you can return an appropriate message
    



def std(lista, valor_excluir = 3):
    # Crear una nueva lista sin el valor a excluir
    lista_filtrada = [x for x in lista if x != valor_excluir]
    
    if len(lista) == 2 and 3 in lista  :
        return 99999
    
    # Si la lista filtrada está vacía o tiene un solo elemento, std no se puede calcular
    if len(lista_filtrada) <= 1:
        return None  # O puedes retornar 0 o un mensaje de error
    
    # Calcular la desviación estándar usando numpy
    std = np.std(lista_filtrada)
    
    return std

def classify_nodule(annotations, threshold = 0.7,lower_limit = 3, upper_limit = 4,three_limit = 1):
    # Fase 1: Contar las ocurrencias de cada anotación
    count_dict = {}
    for annotation in annotations:
        if annotation in count_dict:
            count_dict[annotation] += 1
        else:
            count_dict[annotation] = 1
    
    # Fase 2: Inicializar variables de control
    count_3 = count_dict.get(3, 0)  # Cuántas veces se repite el 3
    benign_others = True  # Solo se repiten 1 y 2
    malignant_others = True  # Solo se repiten 4 y 5
    
    if len(annotations) < lower_limit or len(annotations) > upper_limit : 
        return "Unlabeled","Lenght not valid"
    
    for annotation, count in count_dict.items():
        #print(count)
        if count >= 2:  # Consideramos solo los que se repiten
            
            if annotation == 3:
                if count > three_limit:
                    return "Unlabeled","Demasiados 3" # Si hay más de un 3
            elif annotation in [1, 2]:  # Repeticiones de 1 o 2, posible Benigno
                malignant_others = False  # No puede ser Maligno si hay 1 o 2
            elif annotation in [4, 5]:  # Repeticiones de 4 o 5, posible Maligno
                
                benign_others = False  # No puede ser Benigno si hay 4 o 5
                #print(annotation)
            else:
                # Si hay algo fuera de 1, 2, 4, 5 es Unlabeled
                return "Unlabeled"

    # Fase 3: Verificar desviación estándar
    if std(annotations) > threshold:
        return "Unlabeled",f"Std: {std(annotations)} > {threshold}"

    # Fase 4: Clasificación según las reglas
    if count_3 <= 1:
        
        #print(list(count_dict.keys()))
        if benign_others and (1 in list(count_dict.keys()) or 2 in list(count_dict.keys())):
            #print(list(count_dict.keys()) not in [4,5])
            return "Benigno","benign_others"
        
        elif malignant_others and (4 in list(count_dict.keys()) or 5 in list(count_dict.keys())):
            return "Maligno","malignant_others"

    # Si ninguna de las condiciones se cumple
    return "Unlabeled","None Condition"



def process_and_merge_csv(
        final_path="files/final.csv",
        malignancies_path="files/malignancies.csv",
        output_path="files/SMOTE.csv",
        merge_columns=['patient_id', 'nodule_index'],
        additional_columns=['Malignancy_al'],
        print_diagnostics=True):
    
    # Leer los archivos CSV
    final = pd.read_csv(final_path, index_col=0)
    l_sorted = pd.read_csv(malignancies_path)
    
    # Strip whitespace from column names
    final.columns = final.columns.str.strip()
    l_sorted.columns = l_sorted.columns.str.strip()
    
    # Renombrar columnas en 'final' para que coincidan con 'l_sorted'
    final.rename(columns={'Patient': 'patient_id', 'Nodule': 'nodule_index'}, inplace=True)
    
    # Diagnóstico opcional
    if print_diagnostics:
        print("Columnas en l_sorted:", l_sorted.columns.tolist())
        print("Columnas en final:", final.columns.tolist())
        print("Tipo de dato de 'patient_id' en l_sorted:", l_sorted['patient_id'].dtype)
        print(l_sorted.head())
    
    # Filtrar para obtener las filas de final que están en l_sorted
    filtered_final = final.merge(
        l_sorted[merge_columns + additional_columns],  
        left_on=merge_columns,
        right_on=merge_columns,
        how='inner'
    )
    
    # Diagnóstico opcional
    if print_diagnostics:
        print("Columnas en filtered_final después del merge:", filtered_final.columns.tolist())
        print(filtered_final)
    
    # Guardar el resultado en un archivo CSV
    filtered_final.to_csv(output_path, index=None)
    print(f"Archivo guardado en {output_path}")
    

def treat_final_df(directory = "files/final.csv") : 

    df = pd.read_csv(directory)
    df = df.drop(['Unnamed: 0','Unnamed: 0.1', 'Patient', 'Nodule'], axis=1)
    df.rename(columns={'Malignancy_al':'Malignancy'},inplace=True)
    y_before = df['Malignancy']


    df['Malignancy'] = df['Malignancy'].replace({'Benigno': 0, 'Unlabeled':3, 'Maligno':1})
    return df

def create_train_test_split(df, target_column='Malignancy', test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets and standardize the features.
    
    Parameters:
    df: DataFrame containing the dataset including the target column.
    target_column: The name of the target column.
    test_size: The proportion of the dataset to include in the test split.
    random_state: Controls the shuffling applied to the data before applying the split.
    
    Returns:
    X_train_scaled, X_test_scaled, y_train, y_test: Scaled training and test features, and target labels.
    """
    # Separate features and labels
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test