# sim2real transfer for rare pneumonia lesions detection

We have worked on applying a sim2real transfer appraoch for the detection of rare pneumonia lesions in chest X-rays. 

In order to general abnormal images with the proposed simulator, you'll need to

  1. download normal lung images from [the RSNA dataset](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)

  2. specify the directory path in the variable `normalDir` at `shellScript/simulator.sh`

  3. run the shell script
  
The generated abnormal images, their segmentation masks, and bounding boxes will be saved at `data/sim`, `SegmentationMask/mask1000`, and `simDataInfo/bboxInfo/bboxInfo1000.csv`, respectively. 
