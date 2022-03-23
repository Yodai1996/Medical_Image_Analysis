import pydicom
import csv
from matplotlib import pyplot as plt
%matplotlib inline

# with open('stage_2_train_labels.csv', 'r') as f:
#     read = csv.reader(f)
#     ll = list(read)
# f.close()

with open('stage_2_detailed_class_info.csv', 'r') as f:
    read = csv.reader(f)
    ll = list(read)
f.close()

normalIdList=[]
abnormalUniqueIdList=[]

for i,c in ll[1:]:
    if c=='Normal':
        normalIdList.append(i)
    elif c=='Lung Opacity':
        abnormalIdList.append(i)
        if i not in abnormalUniqueIdList:
            abnormalUniqueIdList.append(i)
            
#save
with open('./normalIdList.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(normalIdList)
f.close()

with open('./abnormalUniqueIdList.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(abnormalUniqueIdList)
f.close()
