import cv2
import os
import time
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import openpyxl
import cv2
#from paddleocr.paddleocr import convert_info_markdown #unused
#from paddleocr import PPStructure #these are unused

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Define the image path
img_path = "docu.jpg"
img_name = os.path.splitext(os.path.basename(img_path))[0]
output_dir = 'output'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

time.sleep(3)
# Read the image
result = ocr.ocr(img_path, cls=True)

# Get the recognized text boxes and content
if result and result[0]:
    # Process OCR results
    ocr_results = result[0]
    
    # Draw annotated image
    image = cv2.imread(img_path)
    boxes = [line[0] for line in ocr_results]
    txts = [line[1][0] for line in ocr_results]
    
    draw_img = draw_ocr(image, boxes, txts, font_path=r'fonts\bookman4.ttf')
    
    # Convert NumPy array to PIL Image and save
    draw_img_pil = Image.fromarray(draw_img)
    draw_img_save_path = os.path.join(output_dir, f"{img_name}_result.jpg")
    draw_img_pil.save(draw_img_save_path)
    print(f"Result image saved to {draw_img_save_path}")
    
    # Manually create Excel file
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OCR Results"
    
    # Add headers
    ws.append(["Box Coordinates", "Recognized Text", "Confidence"])
    
    # Add OCR results to Excel
    for item in ocr_results:
        # Extract box coordinates
        box_coords = str(item[0])
        
        # Extract text and confidence
        text = item[1][0]
        confidence = item[1][1]
        
        # Add row to Excel
        ws.append([box_coords, text, confidence])
    
    # Save Excel file
    excel_path = os.path.join(output_dir, f"{img_name}.xlsx")
    wb.save(excel_path)
    print(f"Excel file saved to {excel_path}")


    #save markdown
    
    try:
        import pandas as pd
        # Read the Excel file
        df = pd.read_excel("output\docu.xlsx")
        
        # Ensure we're working with the 'Recognized Text' column
        text_column = df['Recognized Text']
        
        # Replace newlines with <br> for Markdown compatibility
        text_column = text_column.str.replace('\n', '<br>')
        
        # Convert to Markdown
        markdown_text = " \n"
        for text in text_column:
            # Escape pipe characters to prevent Markdown table formatting issues
            text = text.replace('|', '\\|')
            markdown_text += f" {text} \n"
        #print(markdown_text)
        with open(r'output\recognized_text.md', 'w', encoding='utf-8') as f:
            f.write(markdown_text)

    except Exception as ex:
        print(f"Error converting to Markdown: {ex}")
    
    # Print out the recognized text for reference
    """for idx, text in enumerate(txts, 1):
        print(f"Text {idx}: {text}")"""
else:
    print("No text detected in the image.")