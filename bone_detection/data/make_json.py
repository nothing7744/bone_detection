import SimpleITK as sitk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import traceback
import argparse
import datetime
import json
import cv2
import os
import torchio as tio
import torch


dirname='/data/kehao/myDataset/detection_data4.0'
mapName1='../csv/train_data.csv'
mapName2='../csv/val_data.csv'
mapName3='../csv/test_data.csv'
root='../data_json4.0/'

csvfile_name1= pd.read_csv(mapName1)
csvfile_name1=np.array(csvfile_name1)
csvfile_name2= pd.read_csv(mapName2)
csvfile_name2=np.array(csvfile_name2)
csvfile_name3= pd.read_csv(mapName3)
csvfile_name3=np.array(csvfile_name3)

#我们可以得到新的输出
def make_DataAndJson(csvname,dirname,root_path,phase):
    annotations={}
    annotations["info"] = {
        "description": "bone_lesion bounding box information",
    }
    
    annotations["image"]=[]
    annotations["annotations"]=[]

    
    mydict=dict()
    image=1
    myset=set()

    for i in range(len(csvname)-1):   
        print("====================")
        data=csvname[i]
        source=sitk.ReadImage(data[0])
        source.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))
        origin=source.GetOrigin()
        spacing=source.GetSpacing()
        grapCentCoord=[float(data[2])-origin[0],float(data[3])-origin[1],float(data[4])-origin[2]] 
        # source=sitk.GetArrayFromImage(source)
        source=tio.ScalarImage.from_sitk(source)
        source=np.array(source)
        source=torch.tensor(source).squeeze(0)
        source=np.array(source)
        # print(source)

        # x1,x2,y1,y2,z1,z2=round((grapCentCoord[0]-data[5]/2)/spacing[0]),round((grapCentCoord[0]+data[5]/2)/spacing[0]),\
        # round((grapCentCoord[1]-data[6]/2)/spacing[1]),round((grapCentCoord[1]+data[6]/2)/spacing[1]),\
        # round((grapCentCoord[2]-data[7]/10)/spacing[2]),round((grapCentCoord[2]+data[7]/10)/spacing[2])

        x1,x2,y1,y2,z1,z2=round((grapCentCoord[0]-data[5]/2)/spacing[0]),round((grapCentCoord[0]+data[5]/2)/spacing[0]),\
        round((grapCentCoord[1]-data[6]/2)/spacing[1]),round((grapCentCoord[1]+data[6]/2)/spacing[1]),\
        round((grapCentCoord[2]-data[7]/10)/spacing[2]),round((grapCentCoord[2]+data[7]/10)/spacing[2])
        
        h1,h2=round((grapCentCoord[2]-data[7]/2)/spacing[2]),round((grapCentCoord[2]+data[7]/2)/spacing[2])
        
        for j in range(z1,z2):
            if j<0:
                continue
            # if x1<0 or y1<0 or x2>=source.shape[1] or y2>=source.shape[2] or j>=source.shape[0]:
            #     break
            if x1<0 or y1<0 or x2>=source.shape[0] or y2>=source.shape[1] or j>=source.shape[2]:
                break
            # img=sitk.GetImageFromArray(source[j])
            img=sitk.GetImageFromArray(source[:,:,j])
            filename='image_'+str(10000*int(data[21])+j)+'.mhd'
            annotations["image"].append({"filename":filename,"id":str(10000*int(data[21])+j)})
            annotations["annotations"].append({                        
                            "category_id": 0,             
                            "bbox": [x1, y1, x2, y2],
                             "id": str(10000*(i+1)+j),
                            "image_id":str(10000*int(data[21])+j)
                        })
            sitk.WriteImage(img,os.path.join(dirname,phase)+'/'+filename)
    json_path = os.path.join(root_path, phase+".json")
    with open(json_path, "w") as f:
            json.dump(annotations, f)
            

if __name__=='__main__':
    make_DataAndJson(csvfile_name1,dirname,root,"train")
    make_DataAndJson(csvfile_name2,dirname,root,"val")
    make_DataAndJson(csvfile_name3,dirname,root,"test")
