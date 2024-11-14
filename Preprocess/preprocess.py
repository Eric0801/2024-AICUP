# pip install pdfplumber fitz easyocr tqdm numpy Pillow
import os
import json
import argparse
from tqdm import tqdm
import pdfplumber

import sys
import fitz  
import easyocr
from PIL import Image
import io
import numpy as np
from PIL import UnidentifiedImageError

def load_data(source_path,reader):
    
    """
    從指定的目錄載入參考文件，返回一個字典，其中鍵是檔案名稱，值是提取的文本內容。
    Args:
        source_path (str): 包含PDF文件的目錄路徑。
        reader (easyocr.Reader): 用於從PDF中的圖片提取文字的OCR讀取器。
    Returns:
        Dict[int, str]: 一個字典，鍵是檔案名稱（去掉副檔名），值是文件文本內容。
    """
    
    masked_file_ls = os.listdir(source_path)  
    corpus_dict = {int(file.replace('.pdf', '')): extract_text_from_pdf(os.path.join(source_path, file),reader) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    
    return corpus_dict


def extract_text_from_pdf(pdf_path,reader):
    """
    從PDF文件中提取文字，包括從圖片中通過OCR提取的文字。
    Args:
        pdf_path (str): PDF文件的路徑。
        reader (easyocr.Reader): 用於處理PDF中圖片的OCR讀取器。
    Returns:
        str: 從PDF中提取的完整文本，包括從圖片中OCR提取的文字。
    """
    # 打開PDF文件
    doc = fitz.open(pdf_path)
    extracted_text = ''

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        page_text = page.get_text('text') 
        extracted_text += page_text + '\n'  

        
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0] 
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]  

            try:
                # 嘗試打開圖片
                img = Image.open(io.BytesIO(img_bytes))
                img_np = np.array(img)
                result = reader.readtext(img_np)

                # 將 OCR 结果連接到一起
                for detection in result:
                    text = detection[1]
                    extracted_text += text + '\n'
            except UnidentifiedImageError:
                
                continue  # 跳過無法辨識的圖片
    return extracted_text