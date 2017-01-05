#reference: http://paulbourke.net/dataformats/tiff/
#reference: http://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
#reference: http://www.awaresystems.be/imaging/tiff/faq.html#q3
#reference: https://docs.python.org/2/library/struct.html#format-characters
#reference: https://tools.ietf.org/html/rfc2306

import numpy as np

# see TIFF tags: http://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
TAGS = {'image_width': '0100',
        'image_length': '0101',
        'bits_per_sample': '0102', # short array
        'compression': '0103',
        'photometric': '0106',
        'strip_offsets': '0111', # long
        'orientation': '0112',
        'samples_per_pixel': '0115',
        'rows_per_strip': '0116',
        'strip_byte_count': '0117', # long
        'minimum_sample_value': '0118', # short array
        'maximum_sample_value': '0119', # short array
        'x_resolution': '011a',
        'y_resolution': '011b',
        'planar_configuration': '011c',
        'resolution_unit': '0128',
        'sample_format': '0153'} # short array

# see Python format characters: https://docs.python.org/2/library/struct.html#format-characters
TAGTYPES = {'B': '0001',    # byte
            's': '0002',    # string including c-style null terminating character
            'H': '0003',    # unsigned short (word, 2 bytes)
            'I': '0004'}    # unsigned int (dword, 4 bytes)


def write_tiff(file_name, img, bit_depth=8, shape=None, photometric=2):

    data_bytes = img.size * int(np.ceil(bit_depth / 8.0))
    header_bytes = 8
    footer_bytes = 4
    IFD_count_bytes = 2
    tag_bytes = 12
    num_tags = 14

    if shape is None:
        if len(img.shape) == 1:
            height = img.shape[0]
            width = 1
            channels = 1
        elif len(img.shape) == 2:
            height = img.shape[0]
            width = img.shape[1]
            channels = 1
        else:
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
    else:
        width = shape[1]
        height = shape[0]

    with open(file_name, mode="wb") as f:

        # write header
        f.write(bytearray.fromhex("4d4d"))                                              # big endian
        f.write(bytearray.fromhex("002a"))                                              # TIFF file identifier
        f.write(bytearray.fromhex(int_to_hexstring(data_bytes+header_bytes, 'I', 8)))   # offset to first IFD

        # write the image data
        f.write(img.tobytes())

        # write IFD tags
        f.write(bytearray.fromhex(int_to_hexstring(num_tags, 'H', 4)))                  # number of tags in IFD
        f.write(create_tag_byte_array('image_width', 'H', 1, width))
        f.write(create_tag_byte_array('image_length', 'H', 1, height))
        if channels == 1:
            f.write(create_tag_byte_array('bits_per_sample', 'H', 1, bit_depth))
        else:
            offset = header_bytes+ data_bytes + footer_bytes + IFD_count_bytes + tag_bytes * num_tags
            f.write(create_tag_byte_array('bits_per_sample', 'H', channels, offset, True))
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
            offset += channels * 2
            f.write(create_tag_byte_array('minimum_sample_value', 'H', channels, offset, True))
        if channels == 1:
            f.write(create_tag_byte_array('maximum_sample_value', 'H', 1, 255))
        else:
            offset += channels * 2
            f.write(create_tag_byte_array('maximum_sample_value', 'H', channels, offset, True))
        f.write(create_tag_byte_array('planar_configuration', 'H', 1, 1))
        if channels == 1:
            f.write(create_tag_byte_array('sample_format', 'H', 1, 1))
        else:
            offset += channels * 2
            f.write(create_tag_byte_array('sample_format', 'H', channels, offset, True))
        f.write(bytearray.fromhex("00000000"))  # ending 4 bytes (or offset to next IFD)

        # write IFD array data
        if channels > 1:
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(bit_depth, 'H', 4)))         # bits per channel
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(0, 'H', 4)))                 # minimum value
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(255, 'H', 4)))               # maximum value
            for i in range(channels):
                f.write(bytearray.fromhex(int_to_hexstring(1, 'H', 4)))                 # sample format


def create_tag_byte_array(tagname, type, numValues, data, offset=False):

    tag = TAGS[tagname]
    tag += TAGTYPES[type]
    tag += int_to_hexstring(numValues, 'I', 8)
    if offset:
        tag += int_to_hexstring(data, 'I', 8)
    else:
        tag += int_to_hexstring(data, type, 8)

    return bytearray.fromhex(tag)


def int_to_hexstring(data, data_type='H', str_len=8):

    if data_type in ('B', 'b'):
        fmt1 = '{:0>2}'
    elif data_type in ('H', 'h'):
        fmt1 = '{:0>4}'
    elif data_type in ('I'):
        fmt1 = '{:0>8}'
    else:
        fmt1 = "{:0>4}"
    fmt2 = '{:0<' + str(int(str_len)) + '}'

    hexstring = fmt2.format(fmt1.format(hex(data)[2:]))

    return hexstring


if __name__ == "__main__":

    img = np.zeros((200, 100, 3), dtype=np.uint8)
    img[0,0,:] = 255
    img[-1,-1,:] = 255

    img1 = np.arange(8*8*3, dtype=np.uint8).reshape((8,8,3))
    img2 = np.arange(8*8, dtype=np.uint8).reshape((8,8))
    img3 = np.arange(8, dtype=np.uint8)

    write_tiff("Test.TIF", img2)

