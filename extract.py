# all import statements
import numpy as np
import pandas as pd
import pydicom as pyd
import os
import matplotlib.pyplot as plt
import mudicom
import scipy
import pickle
import cv2
import math
import statistics

from numpy import newaxis
from numpy import array
from os.path import dirname, join
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
from PIL import Image
from scipy.misc import imresize
from scipy.signal import convolve2d
from skimage.segmentation import slic, mark_boundaries, clear_border
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import shannon_entropy
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy

# helper methods

# load_pickle(path_to_pickleFileName): Loads any pickle file saved on the disk.

def load_pickle(fileName):
    with open(fileName, "rb") as fp:
        file = pickle.load(fp)
    return file 

# show(image, title = None) Shows the numpy.ndarray or the image with the title if the title is provided.

def show(img, title=None):
    plt.imshow(img, cmap=plt.cm.bone)
    if title is not None: plt.title = title

# plots(image__list, figsize = (12,6), rows = 2, titles = None): Given images names or numpy.ndarray in a _list_ it will display with a default _figure size_ and _number of rows_ and if given the list of _titles_ with each image it will also display it above the images.

def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap=plt.cm.bone)

# An amatuer way of finding mean of a list containing integers or floats

def mean_from_list(listname):
    listname = pd.Series(listname).fillna(0).tolist()
    counts = 0
    total = 0
    for e in listname:
        if e != 0:
            counts = counts + 1
            total = total + e
    return total/counts


# entropy and glcm features


def s_entropy(image):
    return shannon_entropy(image)

def entropy_simple(image):
    return entropy(image)

class glcm:
    def __init__(self, image, distances = [1, 2, 3], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], properties = ['energy', 'contrast', 'homogeneity', 'correlation']):
        self.image = img_as_ubyte(image.astype('int64'))
        self.distances = distances
        self.angles = angles
        self.glcm_mat = greycomatrix(self.image, distances = self.distances, angles = self.angles, symmetric = True, normed = True)
        self.properties = properties
        
    def correlation(self):
        return greycoprops(self.glcm_mat, 'correlation').flatten()
    
    def homogeneity(self):
        return greycoprops(self.glcm_mat, 'homogeneity').flatten()
    
    def contrast(self):
        return greycoprops(self.glcm_mat, 'contrast').flatten()
    
    def energy(self):
        return greycoprops(self.glcm_mat, 'energy').flatten()
    
    def glcm_all(self):
        return np.hstack([greycoprops(self.glcm_mat, props).ravel() for props in self.properties])
    
        
# region properties

class region_props:
    def __init__(self, image, sq = square(3)):
        self.image = image.astype('int64')
        self.thresh = threshold_otsu(self.image)
        self.bw = closing(self.image > self.thresh, sq)
        self.bw_clear = clear_border(self.bw)
        self.bw_label = label(self.bw_clear)
        self.regions = regionprops(self.bw_label)
        self.lista = []
        for e in self.regions:
            self.lista.append(e.area)
        self.idx = self.lista.index(max(self.lista))
    
    def plot_show_bw(self):
        show(self.bw)
        
    def plot_image(self):
        show(self.bw_clear)
        
    def plt_image_with_label(self):
        show(self.bw_label)
        
    def max_area(self):
        return max(self.lista)
    
    def eccentricity(self):
        return self.regions[self.idx].eccentricity
    
    def euler_number(self):
        return self.regions[self.idx].euler_number
    
    def solidity(self):
        return self.regions[self.idx].solidity
    
    def perimeter(self):
        return self.regions[self.idx].perimeter
    
    def mean_area(self):
        return statistics.mean(self.lista)
    
    def std_area(self):
        return statistics.stdev(self.lista)
    
    def thresh_img(self):
        return self.thresh
    
    def bb(self):
        return self.regions[self.idx].bbox
    
    def bb_area(self):
        return self.regions[self.idx].bbox_area
    
    def centroid_r(self):
        return self.regions[self.idx].centroid
    
    def convex_area_r(self):
        return self.regions[self.idx].convex_area
    
    def coordinates_r(self):
        return self.regions[self.idx].coords
    
    def eq_diameter(self):
        return self.regions[self.idx].equivalent_diameter
    
    def extent_r(self):
        return self.regions[self.idx].extent
    
    def filled_area_r(self):
        return self.regions[self.idx].filled_area
    
    def inertia_tensor_r(self):
        return self.regions[self.idx].inertia_tensor
    
    def inertia_tensor_eigvals_r(self):
        return self.regions[self.idx].inertia_tensor_eigvals
    
    def label_r(self):
        return self.regions[self.idx].label
    
    def local_centroid_r(self):
        return self.regions[self.idx].local_centroid
    
    def maj_ax_len(self):
        return self.regions[self.idx].major_axis_length
    
    def min_ax_len(self):
        return self.regions[self.idx].minor_axis_length
    
    def orient(self):
        return self.regions[self.idx].orientation


# show watershed segmentation

def water_seg(image, footprint = np.ones((3,3))):
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=footprint, labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask = image)
    show(labels)

# moments and hu moments

class moments:
    
    def __init__(self, image):
        self.image = image
        self.moment = cv2.moments(self.image)
        self.hu = cv2.HuMoments(self.moment)
        
    def get_moments(self):
#         keys = [key for key in self.moment.keys()]
        values = [value for value in self.moment.values()]
        return values
    
    def get_HuMoments(self):
        moments_hu = []
        for m in range(len(self.hu)):
            moments_hu.append(self.hu[m][0])
        return moments_hu

