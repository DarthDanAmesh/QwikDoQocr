from paddleocr import PaddleOCR, draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order


import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from paddleocr.ppstructure.recovery.recovery_to_markdown import convert_info_markdown

table_engine = PPStructure(recovery=True)
table_engine = PPStructure(recovery=True, lang='en')

save_folder = './output'
#changed from this #img_path = 'ppstructure/docs/table/1.png'
img_path = 'Sentiment/ocr/fromDocumentation/docu.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

h, w, _ = img.shape
res = sorted_layout_boxes(result, w)
convert_info_markdown(res, save_folder, os.path.basename(img_path).split('.')[0])

#Run the above will install the SLANet_infer.tar and the layout tar
#however the output is: markdown save to ./output\docu_ocr.md below is the markdown
"""
<div align="center">
	<img src="docu/[0, 0, 448, 599]_0.jpg">
</div>
"""

#second trial with different outcome.

"""
#running this will NOT install the SLANet_infer.tar and the layout tar
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = r'Sentiment\ocr\fromDocumentation\docu.jpg'#Sentiment\ocr\fromDocumentation\docu.jpg
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path=r'Sentiment\ocr\fonts\bookman4.ttf')# the relative path is: Sentiment\ocr\fonts\bookman4.ttf
im_show = Image.fromarray(im_show)
im_show.save(r'Sentiment\ocr\fromDocumentation\output\result.jpg')"""

#https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html#python as per the example


#I use this frequently; edit the path appropriately.
"""
img_path = r'Sentiment\ocr\fromDocumentation\docu.jpg'
Sentiment\ocr\fromDocumentation\output\result.jpg
font_path=r'Sentiment\ocr\fonts\bookman4.ttf'
"""