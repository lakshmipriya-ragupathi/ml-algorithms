import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

'''
2. Consider the following images. 
Obtain the histograms for each of the images. 
Using a suitable distance measure (Bhattacharyya Distance) ,
find the distance between the query image and reference images. 
You can use inbuilt function to calculate the Bhattacharyya Distance between two histograms, 
check classroom post for more information.
'''

img1_path = "Query_Image.png"
img2_path = "Reference_Image_1.png"
img3_path = "Reference_Image_2.png"

q_img = cv2.imread(img1_path)
# [rows, columns]
#demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
query_image = q_img[0:200,0:250]
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
plt.imshow(query_image)
plt.show()

r1_img = cv2.imread(img2_path)

# [rows, columns]
reference_image_1 = r1_img[0:200, 0:250]
reference_image_1 = cv2.cvtColor(reference_image_1, cv2.COLOR_BGR2RGB)
plt.imshow(reference_image_1)
plt.show()

r2_img = cv2.imread(img3_path)
# [rows, columns]

reference_image_2 = r2_img[0:200, 0:250]
reference_image_2 = cv2.cvtColor(reference_image_2, cv2.COLOR_BGR2RGB)
plt.imshow(reference_image_2)
plt.show()

#Obtain the histograms for each of the images.


query_hist = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
query_hist_norm = cv2.normalize(query_hist, query_hist).flatten()
plt.plot(query_hist_norm)

ref_1_hist = cv2.calcHist([reference_image_1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
ref_1_norm = cv2.normalize(ref_1_hist, ref_1_hist).flatten()
plt.plot(ref_1_norm)

ref_2_hist = cv2.calcHist([reference_image_2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
ref_2_norm = cv2.normalize(ref_2_hist, ref_2_hist).flatten()
plt.plot(ref_2_norm)

#compute bhattachariya distance:
#battacharyya_distance1 = cv2.compareHist(query_hist_norm, reference_hist1_norm, cv2.HISTCMP_BHATTACHARYYA)
#distance between query image and reference 1
battacharyya_distance1 = cv2.compareHist(query_hist_norm, ref_1_norm, cv2.HISTCMP_BHATTACHARYYA)
print("Bhattacharya distance between query image and reference 1 is : ", battacharyya_distance1)
battacharyya_distance2 = cv2.compareHist(query_hist_norm, ref_2_norm, cv2.HISTCMP_BHATTACHARYYA)
print()
print()
print("Bhattacharya distance between query image and reference 2 is : ", battacharyya_distance2)


