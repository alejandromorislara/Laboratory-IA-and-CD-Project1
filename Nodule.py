import numpy as np
import pylidc as pl
from statistics import median_high,mode
import inspect
import pandas as pd

class AverageNodule():
    def __init__(self, nodule_id, annotation_list,patient_id):
        """
        Inicializa el objeto AverageNodule con su ID y una lista de anotaciones del nódulo.
        
        Parameters:
        - nodule_id: Identificador del nódulo.
        - annotation_list: Lista de objetos de anotaciones de los radiólogos (objetos Annotation).
        """
        self.nodule_id = nodule_id
        self.annotation_list = annotation_list
        self.annotations = dict()
        self.patient_id = patient_id

    def get_methods_info(self):
        """
        Método que devuelve un diccionario con todos los métodos de la clase, donde las claves son los nombres
        de los métodos y los valores son las anotaciones de tipos de los métodos (si las tiene).
        
        Retorna:
        - dict: Diccionario con métodos y sus anotaciones de tipo.
        """
        methods_annotations = {}
        # Obtenemos todos los miembros de la clase que son métodos
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            annotations = inspect.signature(method).return_annotation
            methods_annotations[name] = annotations
        return methods_annotations
    
    
    def get_annot_info(self):
        """
        Calcula las medias de las propiedades de las anotaciones del nódulo.

        Retorna:
        - dict: Diccionario con las medias de volumen, área superficial, malignidad, esfericidad, textura, 
                calcificación, estructura interna, márgenes, espiculaciones, sutileza, diámetros y lobulaciones.
        """
        # Listas para almacenar las propiedades de las anotaciones
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
        
        # Recorremos las anotaciones y recolectamos las propiedades
        for annotation in self.annotation_list:
            sphericities.append(annotation.sphericity)
            volumes.append(annotation.volume)
            surface_areas.append(annotation.surface_area)
            textures.append(annotation.texture)
            malignancies.append(annotation.malignancy)
            calcifications.append(annotation.calcification)  
            internal_structures.append(annotation.internalStructure) 
            margins.append(annotation.margin)  
            spiculations.append(annotation.spiculation)  
            subtleties.append(annotation.subtlety) 
            diameters.append(annotation.diameter) 
            lobulations.append(annotation.lobulation)  
        
        # Calculamos las medias/modas de cada propiedad
        average_data = {
            'mean_volume': np.mean(volumes) if volumes else 0,
            'mean_surface_area': np.mean(surface_areas) if surface_areas else 0,
            'mean_malignancy': median_high(malignancies) if malignancies else 0,
            'mode_sphericity': mode(sphericities) if sphericities else 0,
            'mode_texture': mode(textures) if textures else 0,
            'mode_calcification': mode(calcifications) if calcifications else 0,
            'mode_internal_structure': mode(internal_structures) if internal_structures else 0,
            'mode_margin': mode(margins) if margins else 0,
            'mode_spiculation': mode(spiculations) if spiculations else 0,
            'mode_subtlety': mode(subtleties) if subtleties else 0,
            'mean_diameter': np.mean(diameters) if diameters else 0,
            'mode_lobulation': mode(lobulations) if lobulations else 0
        }
        #print(average_data)
        
        return average_data
    
    def get_x_annotation(self, annotation_number):
        """
        Retorna un diccionario con los datos de una anotación específica del nódulo.

        Parameters:
        - annotation_number: Índice de la anotación en la lista de anotaciones.

        Retorna:
        - dict: Diccionario con las propiedades de la anotación seleccionada (volumen, área superficial, etc.).
        """
        data = dict()
        # Obtener la anotación específica (el índice es annotation_number - 1 ya que los índices en Python son base 0)
        notation = self.annotation_list[int(annotation_number) - 1]
        print(f'Accessing annotation number {annotation_number} from patient')

        # Llenar el diccionario con los valores de las propiedades
        data = {
            'volume': notation.volume,
            'surface_area': notation.surface_area,
            'malignancy': notation.malignancy,
            'sphericity': notation.sphericity,
            'texture': notation.texture,
            'calcification': notation.calcification,
            'internal_structure': notation.internalStructure,
            'margin': notation.margin,
            'spiculation': notation.spiculation,
            'subtlety': notation.subtlety,
            'diameter': notation.diameter,
            'lobulation': notation.lobulation
        }
        
        return data

    
# Base class for managing LIDC queries and scans
class LIDCBase():
    def __init__(self, pid):
        self.pid = pid
        self.scan = None

    def query_scan(self):
        """Query and return the scan for the given patient ID."""
        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == self.pid).first()
        if not self.scan:
            raise ValueError(f"No scan found for patient ID {self.pid}")
        return self.scan


# Derived class PyLIDC with additional functionalities
class PyLIDC(LIDCBase):
    
    def __init__(self, pid):
        super().__init__(pid)

    def get_nodules(self):
        """Return the list of nodules for the queried scan."""
        if not self.scan:
            self.query_scan()  # Ensure the scan is queried
        return self.scan.cluster_annotations()  # This will cluster annotations as nodules via euclidian distance

    def get_nodule_count(self):
        """Return the number of nodules in the scan."""
        nodules = self.get_nodules()
        return len(nodules)
    
    def summary(self):
        """Prints a summary of the scan, including patient ID, number of nodules, and scan dimensions."""
        if not self.scan:
            self.query_scan()

        print(f"Patient ID: {self.pid}")
        print(f"Number of nodules: {self.get_nodule_count()}")
        print(f"Scan dimensions (voxels): {self.scan.to_volume().shape}")
        print(f"Pixel spacing (mm): {self.scan.pixel_spacing}")
        print(f"Slice thickness (mm): {self.scan.slice_thickness}")
        

if __name__ == "__main__" : 

    # Uso del código
    l = PyLIDC('LIDC-IDRI-0003')  # Usamos el paciente número 3 para la demostración
    print(f"Cantidad de nódulos: {l.get_nodule_count()}")  # Muestra el número de nódulos

    print("\nAccediendo a las anotaciones del primer nódulo...")
    nods = l.get_nodules()  # Obtenemos los nódulos
    print(f"Primer nódulo con {len(nods[0])} anotaciones.")

    # Crear un objeto AverageNodule
    nodule = AverageNodule(nodule_id=1, annotation_list=nods[3],patient_id="LIDC-IDRI-0078")
    g = nodule.get_annot_info()  # Mostramos la información de las anotaciones
    
    print(nodule.get_x_annotation(4))
    
    print(f"Moda : {mode([1,2])}") # 1