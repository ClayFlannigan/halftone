#reference: http://paulbourke.net/dataformats/tiff/
#reference: http://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
#reference: http://www.awaresystems.be/imaging/tiff/faq.html#q3
#reference: https://docs.python.org/2/library/struct.html#format-characters
#reference: https://tools.ietf.org/html/rfc2306

import numpy as np
from math import ceil

TAGS = {'image_width':          '0100',     # short
        'image_length':         '0101',     # short
        'bits_per_sample':      '0102',     # short array
        'compression':          '0103',     # short
        'photometric':          '0106',     # short
        'strip_offsets':        '0111',     # long
        'orientation':          '0112',     # short
        'samples_per_pixel':    '0115',     # short
        'rows_per_strip':       '0116',     # short
        'strip_byte_count':     '0117',     # long
        'minimum_sample_value': '0118',     # short array
        'maximum_sample_value': '0119',     # short array
        'x_resolution':         '011a',     # rational
        'y_resolution':         '011b',     # rational
        'planar_configuration': '011c',     # short
        'resolution_unit':      '0128',     # short
        'sample_format':        '0153'}     # short array

TAGTYPES = {'B': '0001',    # byte
            's': '0002',    # ASCII string including c-style null terminating character
            'H': '0003',    # unsigned short (word, 2 bytes)
            'I': '0004',    # unsigned int (2 words, 4 bytes)
            'R': '0005'}    # rational, two ints (4 words, 8 bytes)


def write_tiff(file_name, img, bit_depth=8, photometric=None, DPI=200):
    """
    Write a TIFF image file to disc
        Args:
            file_name (string): full path and file name to write
            img (uint8 numpy array): grayscale or multi-channel image
            bits_depth (int): 1, 2, 4, or 8 bits per channel
            photometric (int): see http://www.awaresystems.be/imaging/tiff/tifftags/photometricinterpretation.html
            DPI (int): pixels per inch
        Returns:
            None
        Raises: IO Error for bad file path
        Affects: Writes over existing files without warning
    """

    print(file_name)

    if len(img.shape) == 1:
        height = 1
        width = img.shape[0]
        channels = 1
    elif len(img.shape) == 2:
        height = img.shape[0]
        width = img.shape[1]
        channels = 1
    else:
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

    if photometric is None:
        if channels == 3:
            photometric = 2  # RGB
        else:
            photometric = 1  # black_is_zero

    pixels_per_byte = 8 // bit_depth
    bytes_per_row = int(ceil(float(width) / pixels_per_byte)) * channels
    data_bytes = bytes_per_row * height
    header_bytes = 8
    footer_bytes = 4
    IFD_count_bytes = 2
    tag_bytes = 12
    num_tags = 17

    with open(file_name, mode="wb") as f:

        # write header
        f.write(bytearray.fromhex("4d4d"))                                              # big endian
        f.write(bytearray.fromhex("002a"))                                              # TIFF file identifier
        f.write(bytearray.fromhex(int_to_hexstring(header_bytes + data_bytes, 'I', 8)))   # offset to first IFD

        # write the image data
        img = flatten_and_pack(img, bit_depth)
        f.write(img.tobytes())

        # write IFD tags
        f.write(bytearray.fromhex(int_to_hexstring(num_tags, 'H', 4)))                  # number of tags in IFD
        f.write(create_tag_byte_array('image_width', 'H', 1, width))
        f.write(create_tag_byte_array('image_length', 'H', 1, height))

        offset = header_bytes + data_bytes + IFD_count_bytes + tag_bytes * num_tags + footer_bytes

        if channels == 1:
            f.write(create_tag_byte_array('bits_per_sample', 'H', 1, bit_depth))
        else:
            f.write(create_tag_byte_array('bits_per_sample', 'H', channels, offset, offset=True))
            offset += channels * 2

        f.write(create_tag_byte_array('compression', 'H', 1, 1))
        f.write(create_tag_byte_array('photometric', 'H', 1, photometric))
        f.write(create_tag_byte_array('strip_offsets', 'I', 1, header_bytes))
        f.write(create_tag_byte_array('orientation', 'H', 1, 1))
        f.write(create_tag_byte_array('samples_per_pixel', 'H', 1, channels))
        f.write(create_tag_byte_array('rows_per_strip', 'H', 1, height))
        f.write(create_tag_byte_array('strip_byte_count', 'I', 1, data_bytes))

        if channels == 1:
            f.write(create_tag_byte_array('minimum_sample_value', 'H', 1, 0))
        else:
            f.write(create_tag_byte_array('minimum_sample_value', 'H', channels, offset, offset=True))
            offset += channels * 2

        if channels == 1:
            f.write(create_tag_byte_array('maximum_sample_value', 'H', 1, 2**bit_depth-1))
        else:
            f.write(create_tag_byte_array('maximum_sample_value', 'H', channels, offset, offset=True))
            offset += channels * 2

        f.write(create_tag_byte_array('x_resolution', 'R', 1, offset, offset=True))
        offset += 8 # for rational type

        f.write(create_tag_byte_array('y_resolution', 'R', 1, offset, offset=True))
        offset += 8  # for rational type

        f.write(create_tag_byte_array('planar_configuration', 'H', 1, 1))
        f.write(create_tag_byte_array('resolution_unit', 'H', 1, 2))

        if channels == 1:
            f.write(create_tag_byte_array('sample_format', 'H', 1, 1))
        else:
            f.write(create_tag_byte_array('sample_format', 'H', channels, offset, offset=True))
            offset += channels * 2

        f.write(bytearray.fromhex("00000000"))  # ending 4 bytes (or offset to next IFD)

        # write IFD array data
        if channels > 1:
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(bit_depth, 'H', 4)))         # bits per channel
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(0, 'H', 4)))                 # minimum value
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(2**bit_depth-1, 'H', 4)))    # maximum value

        x_res_numerator = int_to_hexstring(DPI, 'I', 8)
        x_res_denominator = int_to_hexstring(1, 'I', 8)
        f.write(bytearray.fromhex(x_res_numerator + x_res_denominator))                 # x resolution

        y_res_numerator = int_to_hexstring(DPI, 'I', 8)
        y_res_denominator = int_to_hexstring(1, 'I', 8)
        f.write(bytearray.fromhex(y_res_numerator + y_res_denominator))                 # y resolution

        if channels > 1:
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(1, 'H', 4)))                 # sample format

def flatten_and_pack(img, bits):
    """
    Packs reduced bit depth images into bytes and returns a flattened array
        Args:
            img (uint8 numpy array): grayscale or multi-channel image
            bits (int): 1, 2, 4, or 8 bits per channel
        Returns:
            uint8 numpy array: flattened and packed array
    """

    # pad the image at the end of the rows, so that each row ends on a byte boundary
    pixels_per_byte = 8 // bits
    if len(img.shape) > 1:
        if img.shape[1] % pixels_per_byte != 0:
            img = np.hstack((img, np.zeros((img.shape[0], pixels_per_byte - img.shape[1] % pixels_per_byte), dtype=np.uint8)))

    a = np.right_shift(img, 8-bits)                                             # reduce bit depth
    b = a.flatten()                                                             # flatten
    c = np.zeros(b.size // pixels_per_byte, dtype=np.uint8)
    for i in range(0, pixels_per_byte):
        c += np.left_shift(b[i::pixels_per_byte], (pixels_per_byte-1-i)*bits)   # pack pixels and add to result

    return c

def create_tag_byte_array(tagname, type, numValues, data, offset=False):
    """
    Creates a byte-array for a TIFF tag
        Args:
            tagname (string): one of the enumerated TAGS
            type (int): one of the enumerated TAGTYPES
            numValues (int): number of data values
            data (int): value to write
            offset (bool): True == this is an offset into another part of the file
        Returns:
            byte array of the tag
    """

    tag = TAGS[tagname]
    tag += TAGTYPES[type]
    tag += int_to_hexstring(numValues, 'I', 8)
    if offset:
        tag += int_to_hexstring(data, 'I', 8)
    else:
        tag += int_to_hexstring(data, type, 8)

    return bytearray.fromhex(tag)


def int_to_hexstring(data, data_type='H', str_len=8):
    """
    Takes an integer and creates a hex string of the appropriate length
        Args:
            data (int): value to hexlify
            data_type (int): one of the enumerated TAGTYPES
            str_len (int): number of characters in the resulting string
        Returns:
            hexstring of the data
    """

    if data_type in ('B', 'b'):
        fmt1 = '{:0>2}'
    elif data_type in ('H', 'h'):
        fmt1 = '{:0>4}'
    elif data_type in ('I', 'i'):
        fmt1 = '{:0>8}'
    elif data_type in ('R'):
        fmt1 = '{:0>16}'
    else:
        fmt1 = "{:0>4}"
    fmt2 = '{:0<' + str(int(str_len)) + '}'

    hexstring = fmt2.format(fmt1.format(hex(data)[2:]))

    return hexstring

def test():

    import PIL.Image

    y, x = np.mgrid[0:256, 0:256]
    z = np.ones((256,256)) * 128
    img0 = np.dstack((x, y, z)).astype(np.uint8)
    img1 = y.astype(np.uint8)
    img2 = np.arange(256, dtype=np.uint8)
    img3 = PIL.Image.open("pics/RGB.png")
    img3 = np.array(img3)[:,:,0:3]
    img4 = PIL.Image.open("pics/banff.jpg")
    img4 = np.array(img4)[:,:,0:3]
    img5, _ = (np.mgrid[0:1242, 0:1276] / 1242. * 255.).astype(np.uint8)
    img6, _ = (np.mgrid[0:1007, 0:12] / 1007. * 255.).astype(np.uint8)

    for i in (1, 2, 4, 8):

        write_tiff("Test0_" + str(i) + ".TIF", img0, bit_depth=i)
        write_tiff("Test1_" + str(i) + ".TIF", img1, bit_depth=i)
        write_tiff("Test2_" + str(i) + ".TIF", img2, bit_depth=i)
        write_tiff("Test3_" + str(i) + ".TIF", img3, bit_depth=i)
        write_tiff("Test4_" + str(i) + ".TIF", img4, bit_depth=i)
        write_tiff("Test5_" + str(i) + ".TIF", img5, bit_depth=i)
        write_tiff("Test6_" + str(i) + ".TIF", img6, bit_depth=i)

if __name__ == "__main__":

    test()