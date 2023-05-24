from typing import Union

import numpy as np

from MEDimage.biomarkers.glcm import *
from MEDimage.MEDscan import MEDscan
from MEDimage.processing.discretisation import discretize
from MEDimage.utils.image_volume_obj import image_volume_obj

class glcm_filter():
    """The glcm filter class"""

    def __init__(
                self,
                ndims: int,
                size: int,
                discr_method: str,
                discr_type: str,
                vol_pross: Union[np.ndarray, image_volume_obj],
                n_q: float=None,
                user_set_min_val: float=None,
                dist_correction: Union[bool, str]=False,
                merge_method: str="vol_merge",
                glcm_data: np.ndarray=None
                ):
        
        """The constructor of the glcm filter

        Args:
            ndims (int): Number of dimension of the kernel filter
            size (int): An integer that represent the length along one dimension of local matrice size
            discr_method (str): "global" for applied globally to volume, or "local" for applied to local sub-data matrices.   
            discr_type (str): Discretisaion approach/type must be: "FBS", "FBN", "FBSequal"
                          or "FBNequal".
            n_q (float): Number of bins for FBS algorithm and bin width for FBN algorithm.
            user_set_min_val (float): Minimum of range re-segmentation for FBS discretisation,
                                  for FBN discretisation, this value has no importance as an argument
                                  and will not be used.
            dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                    discretization length difference corrections as used by the `Institute of Physics and
                                                    Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
                                                    Set this variable to false to replicate IBSI results.
                                                    or use string and specify the norm for distance weighting. Weighting is
                                                    only performed if this argument is "manhattan", "euclidean" or "chebyshev".
            vol_pross: union(ndarray, image_volume_object): 3D array fully processed from which the feature inside the kernel is computed
            merge_method (str, optional): merging ''method'' which determines how features are
                                           calculated. One of "average", "slice_merge", "dir_merge" and "vol_merge".
                                           Note that not all combinations of spatial and merge ``method`` are valid.
            glcm_data: a numpy array the same shape as vol_pross containing a full glcm matrix for each voxel, calculated on sub-matrices of size x size x size
                        (leave empty) 
       

        Returns:
            
        """

        assert isinstance(ndims, int) and ndims > 0, "ndims should be a positive integer"
        assert size % 2 == 1 and size > 0, "size should be a positive odd number."
        assert isinstance(discr_method, str) and discr_method in ["global", "local"], "discr_method should be either 'global' or 'local'"

        self.dim = ndims
        self.size = int(size)
        self.vol_pross=vol_pross
        self.merge_method=merge_method
        self.discr_method=discr_method
        self.discr_type=discr_type
        self.user_set_min_val=user_set_min_val
        self.n_q=n_q
        self.dist_correction = dist_correction
        self.glcm_data=glcm_data

    def __glcm_filter_globally(self):
        # Initialize the numpy array with empty dictionaries
        self.glcm_data = np.empty(shape=self.vol_pross.shape, dtype=dict)
        self.glcm_data.fill(np.nan)

        self.user_set_min_val = np.nanmin(self.vol_pross)

        # Bin discretization
        self.vol_pross, bin_n = discretize(
            vol_re=self.vol_pross,
            discr_type=self.discr_type,
            n_q=self.n_q,
            user_set_min_val=self.user_set_min_val,
            ivh=False
            )

        # Get the original shape of the volume
        original_shape = self.vol_pross.shape

        # Pad the volume before filtering
        pad_size = (self.size - 1) // 2
        self.vol_pross = np.pad(self.vol_pross, pad_size, mode="constant", constant_values=np.nan)
        self.glcm_data = np.pad(self.glcm_data, pad_size, mode="constant", constant_values=np.nan)

        # Sub-data initialization: local matrice for local GLCM
        for i in range(pad_size, original_shape[0] + pad_size):
            for j in range(pad_size, original_shape[1] + pad_size):
                for k in range(pad_size, original_shape[2] + pad_size):
                    # Check if voxel value is not = nan
                    if np.isnan(self.vol_pross[i, j, k]):
                        continue

                    # Init submatrix
                    sub_matrix = np.zeros((self.size, self.size), 
                                        dtype=float)

                    # Extract subvolume
                    sub_matrix = self.vol_pross[i - ((self.size-1) // 2): i + ((self.size-1) // 2) + 1,
                                                j - ((self.size-1) // 2): j + ((self.size-1) // 2) + 1,
                                                k - ((self.size-1) // 2): k + ((self.size-1) // 2) + 1]
                    
                    #Compute GLCM from local matrix
                    glcm = extract_all(vol=sub_matrix,
                                    dist_correction=False,
                                    merge_method="vol_merge")
                    
                    # Assign local glcm matrix to corresponding index
                    self.glcm_data[i, j, k] = glcm
                    print(glcm["Fcm_energy"])
            
        # Reverse the padding
        self.vol_pross = self.vol_pross[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
        self.glcm_data = self.glcm_data[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

        return self.glcm_data
        
        
    def __glcm_filter_locally(self):
        # Initialize the numpy array with empty dictionaries
        self.glcm_data = np.empty(shape=self.vol_pross.shape, dtype=dict)
        self.glcm_data.fill(np.nan)

        self.user_set_min_val=np.nanmin(self.vol_pross)
        
        # Get the original shape of the volume
        original_shape = self.vol_pross.shape

        # Pad the volume before filtering
        pad_size = (self.size - 1) // 2
        self.vol_pross = np.pad(self.vol_pross, pad_size, mode="constant", constant_values=np.nan)
        self.glcm_data = np.pad(self.glcm_data, pad_size, mode="constant", constant_values=np.nan)

        # Sub-data initialization: local matrice for local GLCM
        for i in range(pad_size, original_shape[0] + pad_size):
            for j in range(pad_size, original_shape[1] + pad_size):
                for k in range(pad_size, original_shape[2] + pad_size):

                    # Check if voxel value is not = nan
                    if np.isnan(self.vol_pross[i, j, k]):
                        continue

                    # Init submatrix
                    sub_matrix = np.zeros((self.size, self.size), 
                                            dtype=float)

                    # Extract subvolume
                    sub_matrix = self.vol_pross[i - ((self.size-1) // 2): i + ((self.size-1) // 2) + 1,
                                                j - ((self.size-1) // 2): j + ((self.size-1) // 2) + 1,
                                                k - ((self.size-1) // 2): k + ((self.size-1) // 2) + 1]
                    
                    self.user_set_min_val=np.nanmin(sub_matrix)
                    
                    # Bin discretization
                    sub_matrix, bin_n = discretize(
                        vol_re=sub_matrix,
                        discr_type=self.discr_type,
                        n_q=self.n_q,
                        user_set_min_val=self.user_set_min_val,
                        ivh=False
                    )
                    
                    #Compute GLCM from local matrix
                    glcm = extract_all(vol=sub_matrix,
                                        dist_correction=False,
                                        merge_method="vol_merge")
                    
                    # Assign local glcm matrix to corresponding index
                    self.glcm_data[i, j, k] = glcm

        # Reverse the padding
        self.vol_pross = self.vol_pross[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
        self.glcm_data = self.glcm_data[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

        return self.glcm_data

    def filter_glcm(self):
        if self.discr_method == 'local':
            self.__glcm_filter_locally()
        elif self.discr_method == 'global':
            self.__glcm_filter_globally()
        else:
            raise ValueError("discr_method must be 'global' or 'local'")
        return self.glcm_data

def apply_glcm(
            input_images: Union[np.ndarray, image_volume_obj],
            discr_method: str,
            discr_type: str,
            dist_correction: Union[bool, str],
            medscan: MEDscan = None,
            feature_name: str=None,
            n_q: int=None,
            user_set_min_val: float=None,
            ndims: int = 3,
            size: int = 3,
            ) -> np.ndarray:
    """Apply the glcm filter to the input image, 

    Args:
        input_images (ndarray): The images to filter.
        discr_method (str): "global" for applied globally to volume, or "local" for applied to local sub-data matrices.   
        discr_type (str): Discretisaion approach/type must be: "FBS", "FBN", "FBSequal"
                          or "FBNequal".
        dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                      discretization length difference corrections as used by the `Institute of Physics and
                                                      Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
                                                      Set this variable to false to replicate IBSI results.
                                                      Or use string and specify the norm for distance weighting. Weighting is
                                                      only performed if this argument is "manhattan", "euclidean" or "chebyshev".
        medscan (MEDscan, optional): The MEDscan object that will provide the filter parameters.
        feature_name (str, optional): Name of the feature from which kernel is constructed
        n_q (int, optional): Number of bins for FBS algorithm and bin width for FBN algorithm. For FBS, if not defined during function call, will
                             set it to bin_range * (vmax - vmin).
        user_set_min_val (float, optional): Minimum of range re-segmentation for FBS discretisation,
                                            for FBN discretisation, this value has no importance as an argument
                                            and will not be used. No input on this argument will automatically set it as vmin of extracted volume.
        ndims (int, optional): The number of dimensions of the input image.
        size (int, optional): The size of the kernel, which will define the filter kernel dimension.
        orthogonal_rot (bool, optional): If true, the 3D images will be rotated over coronal, axial and sagittal axis.

    Returns:
        ndarray: Image filtered from feature values
    """
    # Check if the input is a numpy array or a Image volume object
    spatial_ref = None
    if type(input_images) == image_volume_obj:
        spatial_ref = input_images.spatialRef
        input_images = input_images.data
    
    # # Convert to shape : (B, W, H, D)
    # input_images = np.expand_dims(input_images.astype(np.float64), axis=0) 
    
    if medscan:
        # Initialize filter class instance
        _filter = glcm_filter(
                ndims=medscan.params.filter.glcm_filter.ndims,
                size=medscan.params.filter.glcm_filter.size,
                discr_method=discr_method,
                discr_type=discr_type,
                n_q=n_q,
                user_set_min_val=user_set_min_val,
                volume=input_images,
                dist_correction=dist_correction
                )

    else:
        # Initialize filter class instance
        _filter = glcm_filter(
                    vol_pross=input_images,
                    ndims=ndims,
                    size=size,
                    discr_method=discr_method,
                    discr_type=discr_type,
                    n_q=n_q,
                    user_set_min_val=user_set_min_val,
                    dist_correction=dist_correction,
                )
        
        glcm_data = _filter.filter_glcm()

        # save the glcm data locally
        np.save(f'glcm_data_{size}_{discr_method}_{discr_type}_{n_q}.npy', glcm_data)


        vol_pross = input_images
        
        for i in range (glcm_data.shape[0]):
            for j in range (glcm_data.shape[1]):
                for k in range (glcm_data.shape[2]):

                    if type(glcm_data[i, j, k]) == float and np.isnan(glcm_data[i, j, k]):
                        continue
                    elif glcm_data[i, j, k][feature_name] is None:
                        vol_pross[i, j, k] = np.nan
                    
                    
                    vol_pross[i, j, k] = glcm_data[i, j, k][feature_name]

    return vol_pross
