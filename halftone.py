#!/usr/bin/env python

from PIL import Image, TiffImagePlugin
import numpy as np
from math import ceil
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter
import argparse
import os, sys, inspect

#import local (modified verion of tifffile)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "tifffile")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import tifffile as tiff


def crop_center(img, new_shape):
    """
    Crop an image equally on each size to create the new_shape
        Args:
            img (numpy array): 2D array to crop
            new_shape: desired shape of the return
        Returns:
            numpy array: array cropped according to shape
    """
    ul = ((img.shape[0]-new_shape[0])/2, (img.shape[1]-new_shape[1])/2)
    br = (ul[0]+new_shape[0], ul[1]+new_shape[1])
    return img[ul[0]:br[0], ul[1]:br[1]]


def gauss_kernel(size, sigma=None, size_y=None, sigma_y=None):
    """
    Generates a 2D Gaussian kernel as a numpy array
        Args:
            size (int): 1/2 the width of the kernel; total width := 2*size+1
            sigma (float): spread of the gaussian in the width direction
            size_y (int): 1/2 the height of the kernel; defaults to size
            sigma_y (float): spread of the gaussian in the height direction; defaults to sigma
        Returns:
            numpy array: normalized 2D gaussian array
    """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    if not sigma:
        sigma = 0.5 * size + .1
    if not sigma_y:
        sigma_y = sigma
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma_y ** 2))
    return g / g.sum()


def resize(a, shape):
    """
    if array a is larger than shape, crop a; if a is smaller than shape, pad a with zeros
        Args:
            a (numpy array): 2D array to resize
            shape: desired shape of the return
        Returns:
            numpy array: array a resized according to shape
    """
    if a.shape[0] < shape[0]:
        a = np.pad(a, ((0, shape[0]-a.shape[0]), (0, 0)), mode="constant")
    if a.shape[1] < shape[1]:
        a = np.pad(a, ((0, 0), (0, shape[1]-a.shape[1])), mode="constant")
    if a.shape[0] > shape[0]:
        a = a[0:shape[0], :]
    if a.shape[1] > shape[1]:
        a = a[:, 0:shape[1]]
    return a


def halftone(cmyk, size, angles, fill, sharpness):
    """
    Generates a halftone image from a cmyk image
        Args:
            cmyk (numpy array): 0.0-1.0 r x c x 4 image
            size (int): half size of the averaging kernel in pixels
            angles (list of float): 4 angles for the relative rotation of each channel
        Returns:
            numpy array: 0.0-1.0 r x c x 4 halftoned image
    """
    halftone_image = np.zeros(cmyk.shape)

    for i, (channel, angle) in enumerate(zip(np.rollaxis(cmyk, 2), angles)):

        # total width of the kernel
        s = 2 * size + 1

        # rotate the image to eliminate overlap between the channels
        rotated = rotate(channel, angle, reshape=True, prefilter=False, order=1)

        # apply a gaussian filter to average over a the region of the kernel
        averaged = gaussian_filter(rotated, size)

        # find the central value of the filtered image; this is the average intensity in the region
        halftone_weights = averaged[size::s, size::s]

        # tile the weight image with the average intensity value
        halftone_weights = np.repeat(np.repeat(halftone_weights, s, 0), s, 1)
        halftone_weights = resize(halftone_weights, rotated.shape)

        # TODO: consider using sigma to scale with magnitude
        # create a 2D gaussian kernel that will be the "dot"; normalize it to be 1.0 in the center
        kernel = gauss_kernel(size, sigma=fill*size)

        # Apply the sharpness multiplier and clip the kernel to 1.0
        kernel *= sharpness / np.max(kernel)
        kernel = np.clip(kernel, 0.0, 1.0)

        # tile the kernel across the image
        num_kernels = np.array(rotated.shape) / s + 1
        tiled_kernel = np.tile(kernel, num_kernels)
        tiled_kernel = resize(tiled_kernel, rotated.shape)

        # multiply the kernel image with the weights to generate the halftone image
        halftone = tiled_kernel * halftone_weights

        # rotate the image back to zero
        halftone = rotate(halftone, -angle, prefilter=False, order=1)

        # crop the image to the original size
        halftone = crop_center(halftone, channel.shape)

        # add this chanel to the full cmyk image
        halftone_image[:,:,i] = halftone

#        Image.fromarray(halftone*255).show()

#    Image.fromarray(cmyk_to_rgb(halftone_image)).show()

    return halftone_image


def cmyk_to_rgb(cmyk):
    """
    Converts a cmyk image to a rgb representation
        Args:
            cmyk (numpy array): 0.0-1.0 r x c x 4 image
        Returns:
            numpy array: 0-255 r x c x 3 image
    """
    rgb = 255 * (1.0 - cmyk[:,:,0:3]) * (1 - np.stack([cmyk[:,:,3],cmyk[:,:,3],cmyk[:,:,3]], axis=2))
    return np.round(rgb).astype(np.uint8)


def rgb_to_cmyk(rgb, percent_gray):
    """
    Converts an rgb image to a cmyk representation
        Args:
            rgb (numpy array): 0-255 r x c x 3 image
            percent_gray (int): 0-100 percent of K channel to replace in CMY
        Returns:
            numpy array: 0.0-1.0 r x c x 4 image
    """
    # TODO: not sure if I should be inverting the values...
    cmy = 1.0 - rgb / 255.0

    k = np.min(cmy, axis=2) * (percent_gray / 100.0)
    k[np.where(np.sum(rgb,axis=2)==0)] = 1.0  # anywhere there is no color, set the k chanel to max
    k_mat = np.stack([k,k,k], axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        cmy = (cmy - k_mat) / (1.0 - k_mat)
        cmy[~np.isfinite(cmy)] = 0.0

    return np.dstack((cmy, k))


def flatten_and_pack(img, bits):
    """
    Converts a float image chanel to a reduced bit depth and flattened representation
        Args:
            img (numpy array): [0-1] r x c x 1 image
            bits (int): [1-8] bit depth of resulting image
        Returns:
            numpy array: [0-255] r * c // (8 // bits) flattened and packed image
    """

    pixels_per_byte = 8//bits
    # final_shape = 2 * np.divide(img.shape, pixels_per_byte).astype(np.int)
    # a = resize(img, final_shape*pixels_per_byte/2)    # crop image to a whole number of bytes
    b = (img * 255).astype(np.uint8)                    # convert from float to 8-bit
    c = np.right_shift(b, 8-bits)                     # reduce bit depth
    d = c.flatten()                                   # flatten

    # pack bits
    e = np.zeros(int(ceil(float(d.size) / pixels_per_byte)), dtype=np.uint8)
    for i in range(0, pixels_per_byte):
        f = np.left_shift(d[i::pixels_per_byte], (pixels_per_byte-1-i)*bits)
        e[0:f.size] += f

#    return np.reshape(e, final_shape)
    return e

def test():

    # test rgb_to_cmyk
    assert np.allclose(rgb_to_cmyk(np.array([[[255, 255, 255]]], dtype=np.uint8), 100), [[[0, 0, 0, 0]]])
    assert np.allclose(rgb_to_cmyk(np.array([[[0, 0, 0]]], dtype=np.uint8), 100), [[[0, 0, 0, 1]]])
    assert np.allclose(rgb_to_cmyk(np.array([[[0, 0, 0]]], dtype=np.uint8), 0), [[[0, 0, 0, 1]]])
    assert np.allclose(rgb_to_cmyk(np.array([[[10, 20, 30]]], dtype=np.uint8), 100), [[[0.66666667, 0.33333333, 0.0, 0.88235294]]])

    # test cmyk_to_rgb
    assert np.allclose(cmyk_to_rgb(np.array([[[0, 0, 0, 1]]])), [[[0, 0, 0]]])
    assert np.allclose(cmyk_to_rgb(np.array([[[1, 1, 1, 0]]])), [[[0, 0, 0]]])
    assert np.allclose(cmyk_to_rgb(np.array([[[0, 0, 0, 0]]])), [[[255, 255, 255]]])
    assert np.allclose(cmyk_to_rgb(np.array([[[0.66666667, 0.33333333, 0.0, 0.88235294]]])), [[[10, 20, 30]]])

    # test inverse relationship between rgb_to_cmyk and cmyk_to_rgb
    for i in range(1000):
        rgb = np.array([[np.random.randint(0, 255, 3)]])
        gray = (np.random.rand(1)*100)[0]
        assert(np.allclose(cmyk_to_rgb(rgb_to_cmyk(rgb, gray)) - rgb, 0.0))


    # TODO: test 2 bit

    # test quantize
    # assert np.allclose(quantize(np.array([[0.0, 1]]), 1), [[255, 0]])
    # assert np.allclose(quantize(np.array([[0.0, .5]]), 1), [[255, 0]])
    # assert np.allclose(quantize(np.array([[0.0, .1]]), 1), [[255, 255]])
    # assert np.allclose(quantize(np.array([[0.0, .01]]), 8), [[255, 252]])
    # assert np.allclose(quantize(np.array([[0.0, 1]]), 3), [[252, 0]])
    # assert np.allclose(quantize(np.array([[0.0, .874]]), 3), [[252, 36]])
    # assert np.allclose(quantize(np.array([[0.0, .875]]), 3), [[252, 0]])
    print flatten_and_pack(np.arange(25).reshape((5,5))/24.0, 2)

if __name__ == '__main__':

    # test = np.zeros((2,2), dtype=np.uint8)
    # test[0,0] = 255
    # test[1,1] = 255
    # tiff.imsave("simple_tifffile.TIF", test)

    test()

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Generates CMYK halftone images from a color image.')
    parser.add_argument("file", type=str, help="input file name")
    parser.add_argument("-a", "--angles", type=int, nargs="+", default = [15, 75, 0, 45], help="four angles for rotation of each channel")
    parser.add_argument("-b", "--bits", type=int, choices=[1, 2, 4, 6, 8], default=8, help="bits of color info per channel")
    parser.add_argument("-c", "--colorize_CMYK", default=False, action="store_true", help="save CMYK files as RGB color images")
    parser.add_argument("-e", "--extra_file_name", type=str, default="_Clr", help="final name addition for each channel")
    parser.add_argument("-f", "--fill", type=float, default=0.5, help="dot fill (size) value")
    parser.add_argument("-g", "--gray", type=int, default=100, help="percent of grey component replacement (K level)")
    parser.add_argument("-l", "--halftone", default=False, action="store_true", help="halftone the image")
    parser.add_argument("-p", "--sharpness", type=float, default=1.0, help="level of sharpness of the dots")
    parser.add_argument("-s", "--size", type=int, default=3, help="half size of averaging region (pixels)")
    args = parser.parse_args()

    # open file
    try:
        im = Image.open(args.file)
    except IOError:
        print "Cannot open ", args.file
        exit(1)

    # convert to numpy array
    img = np.array(im)[:,:,0:3]

    # separate into CMYK channels; might be better to use pyCMS and an ICC color profile
    CMYK = rgb_to_cmyk(img, args.gray)

    # halftone cmyk images
    if  args.halftone:
        CMYK = halftone(CMYK, args.size, args.angles, args.fill, args.sharpness)

    # save files
    # TODO: BITSPERSAMPLE seems to be expecting the im data to be other than uint8
    # TODO: maybe creating a color map is the way to get the bit depth right?
    # TODO: TiffFile seems to do the photometric conversion correct
    f, e = os.path.splitext(args.file)
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[262] = 0 # info[PHOTOMETRIC_INTERPRETATION] = WhiteIsZero
    info[258] = args.bits # info[BITSPERSAMPLE] = args.bits

    for i in range(4):

        filename = f + args.extra_file_name + str(i+1) + ".TIF"

        # save the RGB color version of the individual CMYK images
        if args.colorize_CMYK:
            out = cmyk_to_rgb(CMYK[:,:,i].copy())
            #Image.fromarray(out).save(filename)
            tiff.imsave(filename, out)

        # save the individual CMYK files
        else:
            out = flatten_and_pack(CMYK[:,:,i], args.bits)
            #Image.fromarray(out).save(filename, tiffinfo = info)
            tiff.imsave(filename, out, photometric="miniswhite", bitspersample=args.bits, imageshape=CMYK[:,:,i].shape)

    # save CMYK full image
    Image.fromarray(cmyk_to_rgb(CMYK)).save(f + ".BMP")


