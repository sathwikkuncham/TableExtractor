import utils
import cv2 as cv
import numpy as np
class ImagePreProcessing:
	def GammaAdujst(self,image,gamma=1.5):
        # gamma correction
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction and show the images
		adjusted = cv.LUT(image, table)
		return adjusted
	
	
	def Threshholding(self,image,blockSize=21):
		grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # =====================================================
        # IMAGE FILTERING (using adaptive thresholding)
        # =====================================================
		"""
        ADAPTIVE THRESHOLDING
        Thresholding changes pixels' color values to a specified pixel value if the current pixel value
        is less than a threshold value, which could be:
    
        1. a specified global threshold value provided as an argument to the threshold function (simple thresholding),
        2. the mean value of the pixels in the neighboring area (adaptive thresholding - mean method),
        3. the weighted sum of neigborhood values where the weights are Gaussian windows (adaptive thresholding - Gaussian method).
    
        The last two parameters to the adaptiveThreshold function are the size of the neighboring area and
        the constant C which is subtracted from the mean or weighted mean calculated.
        """
		MAX_THRESHOLD_VALUE = 255
		BLOCK_SIZE = blockSize
		THRESHOLD_CONSTANT = 0
		
		# Filter image
		filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)
		return filtered
		
		
	def StructureExtraction(self,image,sc=10):
		SCALE = sc
		
		# Isolate horizontal and vertical lines using morphological operations
		horizontal = image.copy()
		vertical = image.copy()
		
		horizontal_size = int(horizontal.shape[1] / SCALE)
		horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
		utils.isolate_lines(horizontal, horizontal_structure)
		
		vertical_size = int(vertical.shape[0] / SCALE)
		vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
		utils.isolate_lines(vertical, vertical_structure)
		
		# Create an image mask with just the horizontal
        # and vertical lines in the image. Then find
        # all contours in the mask.
		mask = horizontal + vertical
		
		(contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		
		# Find intersections between the lines
        # to determine if the intersections are table joints.
		intersections = cv.bitwise_and(horizontal, vertical)
		return contours,intersections
		
		
		
	def CropImage(self,image,x1,x2,y1,y2):
		croppedImage = image[y1 :y2,x1:x2]
		return croppedImage

	