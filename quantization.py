import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class QuantizedImage(object):
    def __init__(self, image, n_colors=64, sample_size=1000):

def quantize(image, n_colors=64, sample_size=1000):
    '''
    Parameters:
        n_colors - the number of colors in the quantized image
        sample_size - the number of pixels to sample for kmeans fit
    Input: a numpy-styled image with uint8 rgb
    Output: a numpy-styled image with uint8 rgb
    '''
    float_image = np.array(image, dtype=np.float64) / 255
    w, h, d = image.shape
    assert d == 3
    image_array = np.reshape(image, (w * h, d))
    # randomly sample 1000 pixels
    image_array_sample = shuffle(image_array, random_state=0)[:sample_size]
    # find n_colors clusters for the random sample
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    # predict model on image_array
    # labels map each pixel to a cluster index
    labels = kmeans.predict(image_array)
    # the clusters and their respectie rgb float values
    codebook = kmeans.cluster_centers_
    # compressed_image is the recreation
    # the third dimension should match the dimension of the labels
    compressed_image = np.zeros((w, h, d), dtype=np.uint8)
    # load each pixel map
    label_index = 0
    for i in range(w):
        for j in range(h):
            codebook_label_index = labels[label_index]
            compressed_image[i][j] = codebook[codebook_label_index]
            label_index += 1
    return compressed_image

def test():
    import matplotlib.pyplot as plt
    from scipy.misc import face

    image = face()
    compressed_image = quantize(image, n_colors=300)

    plt.figure(1)
    plt.imshow(image)
    plt.figure(2)
    plt.imshow(compressed_image)
    plt.show()
