import os
import json
import argparse
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具

import sys
import fitz  
import easyocr
from PIL import Image
import io
import numpy as np
from PIL import UnidentifiedImageError

#載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path,reader):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): extract_text_from_pdf(os.path.join(source_path, file),reader) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    
    return corpus_dict


#載入PDF提取文字以及圖片中的文字，並回傳
def extract_text_from_pdf(pdf_path,reader):


    # 打開PDF文件
    doc = fitz.open(pdf_path)
    extracted_text = ''

    # 迴圈遍歷每一頁
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # 提取頁面中的文字
        page_text = page.get_text('text')  # 獲取頁面上的文本
        extracted_text += page_text + '\n'  # 把文本加到最後結果

        # 獲取頁面中的圖片
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]  # 獲取圖片的 xref (引用ID)
            base_image = doc.extract_image(xref)  # 提取圖片
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

'''
# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本
'''