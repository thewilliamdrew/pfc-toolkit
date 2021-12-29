"""
Utilities to generate chunk masks used in the Precomputed Connectome

"""

import os
import shutil
import numpy as np
from tqdm import tqdm, trange
from nilearn import image, input_data
from scipy.spatial.distance import pdist, squareform, cityblock

XMAX = 91
YMAX = 109
ZMAX = 91

chunk_size = 91
num_chunks = 3209
radius = 10

def get_distance_matrix(locations):
    return squareform(pdist(locations, metric='cityblock'))

def get_distances(home_coords, locations):
    return [cityblock(home_coords, voxel) for voxel in locations]

def get_locations(volume):
    return np.transpose(np.stack(np.where(volume)))

def get_extrema(radius_len, home_coords):
    if((home_coords[0]-radius_len)<0):
        xMin = 0
    else:
        xMin = home_coords[0]-radius_len
    if((home_coords[0]+radius_len)>(XMAX-1)):
        xMax = (XMAX-1)
    else:
        xMax = home_coords[0]+radius_len
    if((home_coords[1]-radius_len)<0):
        yMin = 0
    else:
        yMin = home_coords[1]-radius_len
    if((home_coords[1]+radius_len)>(YMAX-1)):
        yMax = (YMAX-1)
    else:
        yMax = home_coords[1]+radius_len
    if((home_coords[2]-radius_len)<0):
        zMin = 0
    else:
        zMin = home_coords[2]-radius_len
    if((home_coords[2]+radius_len)>(ZMAX-1)):
        zMax = (ZMAX-1)
    else:
        zMax = home_coords[2]+radius_len
    return xMin, xMax, yMin, yMax, zMin, zMax

def label_chunk(radius_len, home_coords, voxel_maps, chunk_label):
    # Get boundaries of search area
    xMin, xMax, yMin, yMax, zMin, zMax = get_extrema(radius_len, home_coords)
    # Make a copy of the voxel_map
    search_area = voxel_maps.copy()
    # zero out everything outside the search area, and zero out the home location
    search_area[:xMin,:,:] = 0
    search_area[xMax:,:,:] = 0
    search_area[:,:yMin,:] = 0
    search_area[:,yMax:,:] = 0
    search_area[:,:,:zMin] = 0
    search_area[:,:,zMax:] = 0
    search_area[home_coords[0],home_coords[1],home_coords[2]] = 0
    # Get voxelmap of unassigned voxels inside search area, not including the home location
    search_area = (search_area == -1)
    # Check if there are enough unassigned voxels to fill a chunk
    if(np.sum(search_area)<(chunk_size-1)):
        label_chunk(radius_len+1, home_coords, voxel_maps, chunk_label)
    else:
        # Get coords of unassigned voxels in search area
        locations = get_locations(search_area)
        # Get Manhattan distances of unassigned voxels in search area to home location
#         distances = get_distance_matrix(locations)[0]
        distances = get_distances(home_coords, locations)
        for i in range(chunk_size-1):
            minidx = np.argmin(distances)
            x,y,z = locations[minidx]
            distances[minidx] = 99999
            voxel_maps[x,y,z] = chunk_label
        voxel_maps[home_coords[0],home_coords[1],home_coords[2]] = chunk_label

def get_next_home(voxel_maps):
    # Make a copy of the voxel_map
    search_area = voxel_maps.copy()
    # Get voxelmap of unassigned voxels inside search area
    search_area = (search_area == -1)
    x=0
    y=0
    z=0
    for i in range(ZMAX):
        if(np.sum(search_area[:,:,i])>0):
            z+=i
            break
    for j in range(YMAX):
        if(np.sum(search_area[:,j,z])>0):
            y+=j
            break
    for k in range(XMAX):
        if(np.sum(search_area[k,y,z])>0):
            x+=k
            break
    return x,y,z
    
    