import numpy as np

def dice_coef(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

# def hd95(result, reference, voxelspacing=None, connectivity=1):
#     """
#     95th percentile of the Hausdorff Distance.

#     Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
#     images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
#     commonly used in Biomedical Segmentation challenges.

#     Parameters
#     ----------
#     result : array_like
#         Input data containing objects. Can be any type but will be converted
#         into binary: background where 0, object everywhere else.
#     reference : array_like
#         Input data containing objects. Can be any type but will be converted
#         into binary: background where 0, object everywhere else.
#     voxelspacing : float or sequence of floats, optional
#         The voxelspacing in a distance unit i.e. spacing of elements
#         along each dimension. If a sequence, must be of length equal to
#         the input rank; if a single number, this is used for all axes. If
#         not specified, a grid spacing of unity is implied.
#     connectivity : int
#         The neighbourhood/connectivity considered when determining the surface
#         of the binary objects. This value is passed to
#         `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
#         Note that the connectivity influences the result in the case of the Hausdorff distance.

#     Returns
#     -------
#     hd : float
#         The symmetric Hausdorff Distance between the object(s) in ```result``` and the
#         object(s) in ```reference```. The distance unit is the same as for the spacing of 
#         elements along each dimension, which is usually given in mm.

#     See also
#     --------
#     :func:`hd`

#     Notes
#     -----
#     This is a real metric. The binary images can therefore be supplied in any order.
#     """
#     hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
#     hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
#     hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
#     return hd95

# def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
#     """
#     The distances between the surface voxel of binary objects in result and their
#     nearest partner surface voxel of a binary object in reference.
#     """
#     result = np.atleast_1d(result.astype(np.bool_))
#     reference = np.atleast_1d(reference.astype(np.bool_))
#     if voxelspacing is not None:
#         voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
#         voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
#         if not voxelspacing.flags.contiguous:
#             voxelspacing = voxelspacing.copy()
            
#     # binary structure
#     footprint = generate_binary_structure(result.ndim, connectivity)
    
#     # test for emptiness
#     if 0 == np.count_nonzero(result): 
#         raise RuntimeError('The first supplied array does not contain any binary object.')
#     if 0 == np.count_nonzero(reference): 
#         raise RuntimeError('The second supplied array does not contain any binary object.')    
            
#     # extract only 1-pixel border line of objects
#     result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
#     reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
#     # compute average surface distance        
#     # Note: scipys distance transform is calculated only inside the borders of the
#     #       foreground objects, therefore the input has to be reversed
#     dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
#     sds = dt[result_border]
    
#     return sds