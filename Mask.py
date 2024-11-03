import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path  
from radiomics.featureextractor import *

def load_npy(file_path) : 
    return np.load(file_path)

def numpy_to_sitk(np_array):
    """
    Convierte un arreglo NumPy en un objeto SimpleITK.Image.
    """
    return sitk.GetImageFromArray(np_array)

def extract_radiomics(scan_array, mask_array):
    """
    Extract radiomic features for a specific nodule using pyradiomics.
    Returns:
    Dictionary of radiomic features.
    """
    # Convertir los arrays NumPy a objetos SimpleITK
    scan_sitk = numpy_to_sitk(scan_array)

    # Convertir la máscara booleana (True/False) a una máscara entera (1/0)
    mask_array_int = mask_array.astype(np.uint8)
    mask_sitk = numpy_to_sitk(mask_array_int)

    # Configurar el extractor de radiomics
    extractor = RadiomicsFeatureExtractor()

    # Extraer las características
    features = extractor.execute(scan_sitk, mask_sitk)

    return features


def process_nodule_images_masks(pid, nodules_annotation, vol, consensus_func, calculate_malignancy_func, IMAGE_DIR="data/image", MASK_DIR="data/mask", mask_threshold=8, prefix="prefix"):
    """
    Procesa y guarda las imágenes y máscaras de los nódulos de un paciente, basado en sus anotaciones.

    Parameters:
    - pid: Identificador del paciente.
    - nodules_annotation: Lista de anotaciones de nódulos para el paciente.
    - vol: Array de volumen del pulmón del paciente.
    - consensus_func: Función para calcular el consenso sobre la segmentación del nódulo.
    - calculate_malignancy_func: Función que calcula la malignidad de un nódulo.
    - IMAGE_DIR: Directorio para guardar las imágenes.
    - MASK_DIR: Directorio para guardar las máscaras.
    - mask_threshold: Umbral mínimo de píxeles en una máscara para ser considerado un nódulo válido.
    - prefix: Prefijo para nombrar las imágenes y máscaras.
    """
    # Crear los directorios para almacenar imágenes y máscaras
    patient_image_dir = Path(IMAGE_DIR) / pid
    patient_mask_dir = Path(MASK_DIR) / pid
    Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
    Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

    if len(nodules_annotation) > 0:
        # Pacientes con nódulos
        for nodule_idx, nodule in enumerate(nodules_annotation):
            # Llamada a las imágenes de nódulos. Cada paciente tiene un máximo de 4 anotaciones.
            mask, cbbox, masks = consensus_func(nodule, 0.5, 512)  # Realiza una consolidación del consenso (acuerdo del 50%).
            lung_np_array = vol[cbbox]  # Extrae el volumen del área con el nódulo

            # Calculamos la información de malignidad
            malignancy, cancer_label = calculate_malignancy_func(nodule)

            print(f"Number of nodule slices: {mask.shape[2]}")
            for nodule_slice in range(mask.shape[2]):
                # Filtramos los tamaños de máscara pequeños que pueden interferir en el entrenamiento.
                if np.sum(mask[:, :, nodule_slice]) <= mask_threshold:
                    continue

                # Obtenemos el slice original del volumen del pulmón
                lung_original_slice = lung_np_array[:, :, nodule_slice]
                
                # Verificamos y corregimos valores que puedan causar errores de tipo de datos
                lung_original_slice[lung_original_slice == -0] = 0

                # Nombramos cada archivo: NI = Nodule Image, MA = Mask Original
                nodule_name = "{}_NI{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                mask_name = "{}_MA{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                
                # Guardamos la imagen original y la máscara
                np.save(patient_image_dir / nodule_name, lung_original_slice)
                np.save(patient_mask_dir / mask_name, mask[:, :, nodule_slice])
                
                # Meta información (puedes almacenarla en otro lugar si es necesario)
                meta_list = [pid[-4:], nodule_idx, prefix[nodule_slice], nodule_name, mask_name, malignancy, cancer_label, False]

    print(f"Processing for patient {pid} completed.")