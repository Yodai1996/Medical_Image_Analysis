import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from collections import defaultdict
import csv

df = pd.read_csv("./stage_2_train_labels.csv")
df = df[df["Target"]==1]
df = df.reset_index(drop=True)

# dic = defaultdict(int)
# for v in list(df["patientId"]):
#     dic[v] += 1
    
abnormalSet = set(list(df["patientId"])) #total 6012 entry

with open('filteredData_from_abnormalUniqueIdList.csv', 'r') as f:
    read = csv.reader(f)
    filteredAbnormalData = list(read)[0]
f.close()

#extract small bbox
small_size = 150
results = [[0 for _ in range(4)] for _ in range(4)]

bboxInfo = []
bboxInfo.append(["file", "left1", "top1", "right1", "bottom1", "left2", "top2", "right2", "bottom2", "left3", "top3", "right3", "bottom3", "left4", "top4", "right4", "bottom4"]) 
smallAbnormalSet = set()

for file in filteredAbnormalData:
    v = file[:-4]
    dg = df[df["patientId"]==v]
    dg = dg.reset_index(drop=True)
    n  = len(dg)
    
    exist_small_bbox = False
    
    cnt = 0
    small_cnt = 0
    
    left2, top2, right2, bottom2 = None, None, None, None
    left3, top3, right3, bottom3 = None, None, None, None
    left4, top4, right4, bottom4 = None, None, None, None
    
    for i in range(n):  
        width, height = dg["width"][i], dg["height"][i] 
        
        #search small bboxes
        if width <= small_size and height <= small_size:
            exist_small_bbox = True
            small_cnt += 1
                
        if cnt==0: #the first bbox
            left1, top1, right1, bottom1 = dg["x"][i], dg["y"][i], dg["x"][i]+width, dg["y"][i]+height
        elif cnt == 1: #the second bbox
            left2, top2, right2, bottom2 = dg["x"][i], dg["y"][i], dg["x"][i]+width, dg["y"][i]+height
        elif cnt == 2:
            left3, top3, right3, bottom3 = dg["x"][i], dg["y"][i], dg["x"][i]+width, dg["y"][i]+height
        else:
            left4, top4, right4, bottom4 = dg["x"][i], dg["y"][i], dg["x"][i]+width, dg["y"][i]+height

        cnt += 1
                
    if exist_small_bbox==True:
        smallAbnormalSet.add(file)
        bboxInfo.append([file, left1, top1, right1, bottom1, left2, top2, right2, bottom2, left3, top3, right3, bottom3, left4, top4, right4, bottom4])
        if cnt > 2:
            print("more than 2 bboxes exist in the file {}".format(file))
        if cnt > 3:
            print("more than 3 bboxes exist in the file {}".format(file))
            
#         if cnt > small_cnt:
#             print(cnt, small_cnt)
        print(cnt, small_cnt)
        results[cnt-1][small_cnt-1] += 1
   
#save
with open('exist_under150smallabnormal_bboxInfo.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(bboxInfo)
f.close()    

# #copy data
# saveDir="Original_images_1024pixels"
# for rows in bboxInfo[1:]:
#     file = rows[0]
#     img = Image.open("./data/AbnormalDir/" + file)
#     img.save("./data/" + saveDir + "/" + file)
