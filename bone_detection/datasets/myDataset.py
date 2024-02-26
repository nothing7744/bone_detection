from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import json
import time
import numpy as np
import copy
import itertools
from collections import defaultdict
import sys
import json
import time
from pathlib import Path
import datasets.transforms as T
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

import SimpleITK as sitk
import torchio as tio


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def window_transform(ct_array, windowWidth, windowCenter, normal=True):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newing = (ct_array - minWindow)/float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1
    #将值域转到0-255之间,例如要看头颅时， 我们只需将头颅的值域转换到 0-255 就行了
    if not normal:
        newing = (newing *255).astype('uint8')
    return newing

class BoneDetection():
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        # super(BoneDetection, self).__init__(root, transforms, transform, target_transform)
        self.dataset =Bone_lessions(annFile)
        # print(type(self.dataset))
        # print((self.dataset).shape) 
        # self.ids = list(sorted(self.dataset.imgs.keys()))
        self.ids = list(sorted(self.dataset.imgs))
        self.root=root
        self.__transforms=transforms

        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)
        # print(len(self.ids))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        dataset = self.dataset
        img_id = self.ids[index]
        # print(img_id)
        # id=img_id
        # print(type(img_id))
        # print(img_id)
        # print(type(img_id))
        # print(img_id)
        id=dataset.getAnnoIds(imgIds=img_id)
        # print(type(id))
        # print(id)
        id=[int(a) for a in id]
        # print(id)
        # target = dataset.loadAnns(id)
        target = dataset.loadAnns(id)
        path = dataset.loadImgs(img_id)[0]['filename']
        # print("???????????????????????")
        # print(path)
        # sitk.ReadImage(data[0])
        #这里得到image_path,和target,最后和dataloader一起进行测试
        #可以在这里对img进行处理
        img = sitk.ReadImage(os.path.join(self.root, path))
        img.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))
        img=tio.ScalarImage.from_sitk(img)
        #we need add
        img=np.array(img)
        img=window_transform(img,1500,300)
        img= torch.as_tensor(img, dtype=torch.float32)
        # print("===============1========================")
        # print(img>torch.tensor([[[0.1]]]))
        # print(img[0,0,0])
        img=img.squeeze(3)
        img=img.repeat(3,1,1)
    #################### mytarget ################################
        mytarget={}
        box = [obj["bbox"] for obj in target]
        bounding_box = [int(obj["bounding box"]) for obj in target]
        bounding_box = torch.tensor(bounding_box, dtype=torch.int64)
        mytarget["bounding_box"]=bounding_box
        boxes = torch.as_tensor(box, dtype=torch.float32).reshape(-1, 4)
        # classes = [int(obj["category_id"])+1 for obj in target]
        classes = [int(obj["category_id"])+2 if obj["bbox"][0]==0 or obj["bbox"][2]==0 else int(obj["category_id"])+1 for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)
        mytarget["boxes"]=boxes
        mytarget["labels"]=classes
        mytarget["orig_size"]=torch.as_tensor([512,512])
        mytarget['image_id']=torch.tensor([int(img_id)])
        
        # #对得到的数据进行数据增强(新加入的代码部分)
        # print(img.shape)
        img=img.squeeze(0).permute(1,2,0)
        img=img.numpy()
        img=(img*255).astype(np.uint8)
        img=Image.fromarray(img)
        if self.__transforms is not None:
            img, mytarget = self.__transforms(img, mytarget)
        return img,mytarget

    def __len__(self):
        return len(self.ids)
    

class Bone_lessions:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                print("json file")
                dataset = json.load(f)
                # print(type(dataset))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()


    def createIndex(self):
        # create index
        print('creating index...')
        anns, imgs = {}, {}
        imgtoAnns=defaultdict(list)
        if 'annotations' in self.dataset:
            print("hello,world")
            for ann in self.dataset['annotations']:
                # print("gggggggggggggggggggggggggg")
                # print(type(ann))
                anns[ann['id']] = ann
                imgtoAnns[ann['image_id']].append(ann)
                # print(imgtoAnns)       
        if 'image' in self.dataset:
            for img in self.dataset['image']:
                imgs[img['id']] = img
        print('index created!')
        # create class members
        self.anns = anns
        self.imgs = imgs
        self.imgToAnns=imgtoAnns
    

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)
    
    def getAnnoIds(self,imgIds=[]):
        # imgIds=imgIds if _isArrayLike(imgIds) else [imgIds]
        imgIds=[imgIds]
        # print(imgIds)
        # print("===========================")
        # print(type(imgIds))

        # print("self.imgToAnns")
        # print(self.imgToAnns)
        lists=[self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]

        # print("+++++++++++++++++++++++++++")
        # print(lists)
        annos=list(itertools.chain.from_iterable(lists))
        # print("---------------------------")
        # print(annos)
        ids=[ann['id'] for ann in annos]
        return ids

    def getCatIds(self):
        ids=[]
        return ids


    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            # print("isArrayLike")
            return [self.anns[str(id)] for id in ids]
        else:
            return [self.anns[str(ids)]]
        # return [self.anns[ids]]


    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        # if _isArrayLike(ids):
        #     return [self.imgs[id] for id in ids]
        # else:
        return [self.imgs[ids]]


def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [412,462, 512, 562, 612]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=800),
            #     T.RandomResize([400, 500, 600]),
            # ),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([512], max_size=800),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_bone(image_set,args):
    if image_set=='train':
        # dataset = BoneDetection(args.train_root,annFile=args.train_anno,transforms=None)
        dataset = BoneDetection(args.train_root,annFile=args.train_anno,transforms=make_transforms(image_set))
    else:
        # dataset = BoneDetection(args.val_root,annFile=args.val_anno)
        dataset = BoneDetection(args.val_root,annFile=args.val_anno,transforms=make_transforms(image_set))
    return dataset
                


