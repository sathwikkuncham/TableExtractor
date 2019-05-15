#!/usr/bin/python
# -*- coding: utf-8 -*-
from TableStructure import TableStructure
import cv2 as cv
import numpy as np
import utils as tableutils
import flask
import ImagePreProcessing as ipp
#import fitz
import urllib.request
from flask import request, jsonify
import json
from azure.storage.blob import BlockBlobService
import os

app = flask.Flask(__name__)
app.config['DEBUG'] = True


class TableExtraction:

    def setXValue(self, table, n):
        x1 = table[n].x
        x2 = table[n].x + table[n].w
        return (x1, x2)

    def setYValue(self, table, n):
        y1 = table[n].y
        y2 = table[n].y + table[n].h
        return (y1, y2)

    def tableSort(self, tables):
        tables.sort(key=lambda obj: (obj.y, obj.y + obj.h, obj.x, obj.x + obj.w))
        return tables

    def ExtractTable(self, image, table):
        te = TableExtraction()
        ippObj = ipp.ImagePreProcessing()
        tableSize = len(table)
        (x1, x2) = te.setXValue(table, 0)
        (y1, y2) = te.setYValue(table, 0)
        print(tableSize)
        if tableSize == 2:
            (x1, x2) = te.setXValue(table, 1)
            y1 = y2
            y2 = table[1].y + table[1].h
        elif tableSize > 2:
            x1 = min(table[tbl].x for tbl in range(1, len(table)))
            x2 = max(table[tbl].x + table[tbl].w for tbl in range(1,len(table)))
            y1 = y2
            y2 = max(table[tbl].x + table[tbl].w for tbl in range(2,len(table)))

        images = ippObj.CropImage(image, x1, x2, y1, y2)
        print(type(images))
        return images


    @app.route('/table', methods=['POST'])
    def DetectTable():
        content = request.json
        download_url = "https://tableextractor.blob.core.windows.net/extracted-images/1557478921912_Page%2001.png"
        response = urllib.request.urlopen(download_url)
        txt=download_url.split('/')
        imageName=txt[len(txt)-1]
        file = open("ExtracedTables\\"+imageName, 'wb')
        file.write(response.read())
        file.close()
        image=cv.imread("ExtracedTables\\"+imageName,1)
        imageCopy = image
        ippObj = ipp.ImagePreProcessing()
        image = ippObj.GammaAdujst(image)
        image = ippObj.Threshholding(image, 21)
        (contours, intersections) = ippObj.StructureExtraction(image,10)

        # Get tables from the images

        tables = []  # list of tables
        for i in range(len(contours)):
            (rect, table_joints) = tableutils.verify_table(contours[i],intersections)
            if rect == None or table_joints == None:
                continue

            # Create a new instance of a table

            table = TableStructure(rect[0], rect[1], rect[2], rect[3])

            # Get an n-dimensional array of the coordinates of the table joints

            joint_coords = []
            for i in range(len(table_joints)):
                joint_coords.append(table_joints[i][0][0])

            joint_coords = np.asarray(joint_coords)

            # Returns indices of coordinates in sorted order
            # Sorts based on parameters (aka keys) starting from the last parameter, then second-to-last, etc

            sorted_indices = np.lexsort((joint_coords[:, 0],joint_coords[:, 1]))
            joint_coords = joint_coords[sorted_indices]

            # Store joint coordinates in the table instance

            table.set_joints(joint_coords)
            tables.append(table)    
        te = TableExtraction()
        tables = te.tableSort(tables)
        images = te.ExtractTable(imageCopy, tables)
        cv.imwrite("ExtracedTables/"+imageName, images)
        result=te.StoreExtractedTable(images)
        return result
		
		
    def StoreExtractedTable(self,images):
        block_blob_service = BlockBlobService(account_name='tableextractor',account_key='OA15EYbHBnD3X+p17r30L59gZOVV91Lht5Y0tLidf/xdVexI0UVKoy8Z+/mYX+cFepSVMSElZeIrLYznZ22y2A==  ')
        container_name = 'extracted-table-images'
        block_blob_service.create_container(container_name)
        # local_path = '/home/vikas/NeuroNer/NeuroNER-master/src/BlobStorage/'
        local_path = "C:\\Users\\M1049308\\Desktop\\tableExtract\\ExtracedTables"
        for files in os.listdir(local_path):
            block_blob_service.create_blob_from_path(container_name,files,os.path.join(local_path, files))
            os.remove(os.path.join(local_path, files))

        print('\nList blobs in the container')
        generator = block_blob_service.list_blobs(container_name)
        print(generator)
        ImageNames = []
        for blob in generator:
            print('\t Blob name: ' + blob.name)
            ImageNames.append(blob.name)
        
        return jsonify(BlobNames=ImageNames)

    @app.route('/ImageExtractor', methods=['POST'])
    def ImageExtractor():
        content = request.json
        download_url = content['body']
        response = urllib.request.urlopen(download_url)
        txt=download_url.split('/')
        pdfname=txt[len(txt)-1]
        x=pdfname.split('.')
        file = open('document.pdf', 'wb')
        file.write(response.read())
        file.close()
        print('Completed')
        doc = fitz.open('document.pdf')
        cnt = 0
        for i in range(len(doc)):
            imglist = doc.getPageImageList(i)
            for img in imglist:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:
                    if(i>=0 and i<=8 ):
                        pix.writePNG('/home/vikas/NeuroNer/NeuroNER-master/src/BlobStorage/'+x[0]+'_Page 0%s.png' % str(i+1))
                    else:
                        pix.writePNG('/home/vikas/NeuroNer/NeuroNER-master/src/BlobStorage/'+x[0]+'_Page %s.png' % str(i+1))
                else:

                    fitz.Pixmap(fitz.csRGB, pix)
                    if(i>=0 and i<=8 ):
                        pix.writePNG('/home/vikas/NeuroNer/NeuroNER-master/src/BlobStorage/'+x[0]+'_Page 0%s.png' % str(i+1))
                    else:
                        pix.writePNG('/home/vikas/NeuroNer/NeuroNER-master/src/BlobStorage/'+x[0]+'_Page %s.png' % str(i+1))



                    cnt = cnt + 1
                    pix1 = None
            pix = None

        block_blob_service = BlockBlobService(account_name='tableextractor',account_key='OA15EYbHBnD3X+p17r30L59gZOVV91Lht5Y0tLidf/xdVexI0UVKoy8Z+/mYX+cFepSVMSElZeIrLYznZ22y2A==  ')
        container_name = 'extracted-images'
        block_blob_service.create_container(container_name)
        local_path = '/home/vikas/NeuroNer/NeuroNER-master/src/BlobStorage/'
        for files in os.listdir(local_path):
            block_blob_service.create_blob_from_path(container_name,files,os.path.join(local_path, files))
            os.remove(os.path.join(local_path, files))

        print('\nList blobs in the container')
        generator = block_blob_service.list_blobs(container_name)
        print(generator)
        ImageNames = []
        for blob in generator:
            if(x[0] in blob.name):
                print('\t Blob name: ' + blob.name)
                ImageNames.append(blob.name)


        return jsonify(BlobNames=ImageNames)

    @app.route("/")
        def hello():
            return "Hello World!"



app.run()
