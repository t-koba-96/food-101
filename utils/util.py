import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import cv2


def imshape(image):
    image=image/2+0.5
    npimg=image.numpy()
    return np.transpose(npimg,(1,2,0))



def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show



def normalize_heatmap(x):
    min = x.min()
    max = x.max()
    result = (x-min)/(max-min)
    return result


def show_predict(images,labels,classes,net,device,image_num,batch):
   
   images_gpu = images.to(device)    
   outputs=net(images_gpu)
   _,predicted=torch.max(outputs,1)
   predicted=predicted.cpu()
   image_batch=images[image_num:image_num+batch,:]
   # 格子状に表示
   imshow(vutils.make_grid(image_batch,nrow=4,padding=1))
   print('GroundTruth: ', ', '.join('%5s' % classes[labels[j]] for j in range(batch))) 
   print('Predict    : ', ', '.join('%5s' % classes[predicted[j]] for j in range(batch)))


def show_attention(images,net,device,image_num):
   images_gpu = images.to(device)    
   at_outputs=net(images_gpu)
   at_predicted=at_outputs.cpu()
   attention=at_predicted.detach()
   
   img=imshape(images[image_num,:])

   #attention map
   heatmap = attention[image_num,:,:,:]
   heatmap = heatmap.numpy()
   heatmap = np.average(heatmap,axis=0)
   heatmap = normalize_heatmap(heatmap)
   # 元の画像と同じサイズになるようにヒートマップのサイズを変更
   heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
   #特徴ベクトルを256スケール化
   heatmap = np.uint8(255 * heatmap)
   # RGBに変更
   heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
   #戻す
   heatmap=heatmap/255
   # 0.5はヒートマップの強度係数
   s_img = heatmap * 0.5 + img

   #plt
   image_list=[img,heatmap,s_img]
   fig=plt.figure(figsize=(10, 10))
   for i,data in enumerate(image_list):
      fig.add_subplot(1, 3, i+1)
      plt.imshow(data)
   plt.show()
