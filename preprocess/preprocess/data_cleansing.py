import pydicom
from matplotlib import pyplot as plt
%matplotlib inline

import csv
from PIL import Image

#import csv
with open('normalIdList.csv', 'r') as f:
    read = csv.reader(f)
    normalIdList = list(read)[0]
f.close()

with open('abnormalUniqueIdList.csv', 'r') as f:
    read = csv.reader(f)
    abnormalUniqueIdList = list(read)[0]
f.close()

#data cleasning
filtered_indices_abnormal=[]
for file in abnormalUniqueIdList:
    d = pydicom.read_file('data/stage_2_train_images/' + file + '.dcm')
    
    #age >= 18 & Position = PA
    if int(d.PatientAge) >= 18 and d.ViewPosition=="PA":
        filtered_indices_abnormal.append(file + ".png")
        
filtered_indices_normal=[]
for file in normalIdList:
    d = pydicom.read_file('data/stage_2_train_images/' + file + '.dcm')
    
    #age >= 18 & Position = PA
    if int(d.PatientAge) >= 18 and d.ViewPosition=="PA":
        filtered_indices_normal.append(file + ".png")     
        
#save 
with open('./filteredData_from_abnormalUniqueIdList.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(filtered_indices_abnormal)
f.close()

with open('./filteredData_from_normalIdList.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(filtered_indices_normal)
f.close()
