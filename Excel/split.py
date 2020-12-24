# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:24:29 2019

@author: Rhodia
"""

import xlrd
import pandas as pd
import os
from openpyxl import load_workbook
os.chdir("C:/Users/Rhodia/Desktop/Task")

excel_name = "try.xlsx"

wb = xlrd.open_workbook(excel_name)
sheets = wb.sheet_names()
df = pd.DataFrame()
xl = pd.ExcelFile(excel_name)
sheet_name = xl.sheet_names

# zhiju or luodizhiju
skiprows = [4,4,1,3,3,3,3,3,3,4,1,3,3,4,4]

total_list = []
for i in range(1,len(sheets)):
    data = pd.read_excel(excel_name, skiprows = skiprows[i-1], sheet_name=sheet_name[i],  index=False, encoding='gbk',header=None)
    area_list = list(set(data.iloc[:,0]))
    for i in area_list:
        if i not in total_list:
            total_list.append(i)

for i in total_list:
    df = pd.DataFrame()
    filename = "畅享翼夏完成情况_%s.xlsx" % i
    df.to_excel(filename,sheet_name="sheet17",index=False,encoding='gbk',header=None)
    

for i in range(1,len(sheets)):
    data = pd.read_excel(excel_name, skiprows = skiprows[i-1], sheet_name=sheet_name[i],  index=False, encoding='gbk',header=None)
    area_list = list(set(data.iloc[:,0]))
    for j in area_list:
        head_lines = pd.read_excel(excel_name, sheet_name=sheet_name[i],index=False,encoding='gbk',header=None)
        head_lines = head_lines[:skiprows[i-1]]
        df_data = data[data.iloc[:,0]==j]
        df = pd.concat([head_lines,df_data],axis=0)
        new_excel_name = "畅享翼夏完成情况_%s.xlsx" % j
        book = load_workbook(new_excel_name)
        writer = pd.ExcelWriter(new_excel_name, engine='openpyxl')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        child_sheet_name = sheet_name[i]
        df.to_excel(writer, sheet_name=child_sheet_name, index=False,encoding='gbk',header=None)
        writer.save()
        writer.close()   