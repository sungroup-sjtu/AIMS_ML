from cairosvg import svg2png
import os

svg_files = list(os.listdir('./'))
svg_files.remove('png')
svg_files.remove('svg2png.py')
for i in svg_files:
    out_file = './png/'+ i.split('.')[0]+'.png'
    print(out_file)
    svg2png(url=i, write_to=out_file)