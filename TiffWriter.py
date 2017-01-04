#reference: http://paulbourke.net/dataformats/tiff/
#reference: http://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
#reference: http://www.awaresystems.be/imaging/tiff/faq.html#q3
#reference: https://docs.python.org/2/library/struct.html#format-characters
#reference: https://tools.ietf.org/html/rfc2306

'''
Baseline TIFF per[TIFF] requires that the following fields be present for all BiLevel Images:
ImageWidth, ImageLength, Compression, PhotometricInterpretation, StripOffsets, RowsPerStrip,
StripByteCounts, XResolution, YResolution and ResolutionUnit.
'''

import numpy as np

# see TIFF tags: http://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
TAGS = {'image_width': '0100',
        'image_length': '0101',
        'bits_per_sample': '0102', # array
        'compression': '0103',
        'photometric': '0106',
        'strip_offsets': '0111', # long
        'orientation': '0112',
        'samples_per_pixel': '0115',
        'rows_per_strip': '0116',
        'strip_byte_counts': '0117', # long
        'minimum_sample_value': '0018', # array
        'maximum_sample_value': '0019', # array
        'x_resolution': '011a',
        'y_resolution': '011b',
        'planar_configuration': '011c',
        'resolution_unit': '0128',
        'sample_format': '0153'} # array

# see Python format characters: https://docs.python.org/2/library/struct.html#format-characters
TAGTYPES = {'B': '0100',    # byte
            's': '0200',    # string including c-style null terminating character
            'H': '0300',    # unsigned short (word, 2 bytes)
            'I': '0400'}    # unsigned int (dword, 4 bytes)

def writeTIFF(fileName, img, bitDepth=8, shape=None, photometric=0):

    with open(fileName, mode="wb") as f:

        # write header
        f.write(bytearray.fromhex("4949"))      # little endian
        f.write(bytearray.fromhex("2a00"))      # TIFF file identifier
        f.write(bytearray.fromhex("08000000"))  # offset to IFD

        # write IFD
        if shape is None:
            if len(img.shape) == 1:
                width = 1
                length = img.shape[0]
            elif len(img.shape) == 2:
                width = img.shape[1]
                length = img.shape[0]
            else:
                raise Exception("Multichannel Images Not Supported")
        else:
            width = shape[1]
            length = shape[0]
        f.write(bytearray.fromhex("0400"))      # number of tags in IFD
        f.write(createTagByteArray('image_width', 'H', 1, width))
        f.write(createTagByteArray('image_length', 'H', 1, length))
        f.write(createTagByteArray('bits_per_sample', 'H', 1, bitDepth))
        f.write(createTagByteArray('photometric', 'H', 1, photometric))
        f.write(bytearray.fromhex("00000000"))  # ending 4 bytes (or offset to next IFD)

        # write the image data
        f.write(img.tobytes())


def createTagByteArray(tagname, type, numValues, data):

    tag = TAGS[tagname]
    tag += TAGTYPES[type]
    tag += int_to_hexstring_with_endian_swap(numValues)
    tag += int_to_hexstring_with_endian_swap(data)

    print tag

    return bytearray.fromhex(tag)


# swap endianness of int and convert to hex string
def int_to_hexstring_with_endian_swap(data, bits=16):
    a = np.array(data, dtype=np.uint16)
    b = a.byteswap()
    c = hex(b)[2:]
    if len(c) % 2 == 1:
        c = '0' + c
    if bits == 24:
        d = '{:0<12}'.format(c)
    else:
        d = '{:0<8}'.format(c)
    return d

if __name__ == "__main__":

    #TODO: this is broken.  Consider using big endian instead
    print int_to_hexstring_with_endian_swap(256, 16)

    writeTIFF("Test.TIF", np.arange(256, dtype=np.uint8), shape=(16,16))

