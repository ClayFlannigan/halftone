# python-halftone

A python module that uses creates CMYK images and halftone repre

Heavily adapted from [this StackOverflow answer][so] by [fraxel][fr] and [this code][gh] by [Phil Gyford][pg]

[pil]: http://www.pythonware.com/products/pil/
[so]: http://stackoverflow.com/questions/10572274/halftone-images-in-python/10575940#10575940
[fr]: http://stackoverflow.com/users/1175101/fraxel
[gh]: https://github.com/philgyford/python-halftone
[pg]: https://github.com/philgyford

## Usage

    ./haltone.py "filename"

Creates four CMYK images and a combined image.  Provides options for haltoning or not (-d)


| OPTION 												| DESCRIPTION |
| ----------------------------------------------------- | ----------- |
| -h, --help            								| show this help message and exit |
| -b {1,2,4,6,8}, --bits {1,2,4,6,8}					| bits of color info per channel |
| -s SIZE, --size SIZE  								| half size of averaging region (pixels) |
| -f FILL, --fill FILL  								| dot fill (size) value |
| -a ANGLES [ANGLES ...], --angles ANGLES [ANGLES ...]	| four angles for rotation of each channel |
| -g GRAY, --gray GRAY  								| percent of grey component replacement (K level) |
| -d, --do_not_halftone									| don't do halftoning |
| -e EXTRA_FILE_NAME, --extra_file_name EXTRA_FILE_NAME | final name addition for each channel |

