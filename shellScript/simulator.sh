#!/bin/bash

num=1000 #you can also specify 200 or 30
normalDir="MODIFY HERE: SPECIFY NORMAL DIR" #download normal lung images from the RSNA dataset and specify the directory
normalIdList="Medical_Image_Analysis/annotation_info/normalFiles${num}.csv"

abnormalDir="Medical_Image_Analysis/data/sim${m}/"
segMaskDir="Medical_Image_Analysis/SegmentationMask/mask${m}/"
saveParaPath="Medical_Image_Analysis/simDataInfo/paraInfo/parameterInfo${m}.csv"
saveBboxPath="Medical_Image_Analysis/simDataInfo/bboxInfo/bboxInfo${m}.csv"

mkdir -p ${abnormalDir}
mkdir -p ${segMaskDir}
mkdir -p "/Medical_Image_Analysis/simDataInfo/paraInfo/"
mkdir -p "/Medical_Image_Analysis/simDataInfo/bboxInfo/"

# you can change the following values as you like
res=2
persistence='0.8'
lacunarity=4
scale='0.5'
smoothArea='0.4'

pipenv run python ../WisteriaCodes/fractalGenerator.py ${normalIdList} ${normalDir} ${abnormalDir} ${segMaskDir} ${saveParaPath} ${saveBboxPath} ${res} ${persistence} ${lacunarity} ${scale} ${smoothArea}
