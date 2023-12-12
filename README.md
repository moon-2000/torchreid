# torchreid

## OSNET model on Market1501 Image dataset   
### For Training    
1- Download the Market1501 dataset from https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html   
2- Make sure to extract the dataset in market1501 foler.         
3- Clone deep-person-reid repo from https://github.com/KaiyangZhou/deep-person-reid.git   
4- Comment line 29 in deep-person-reid/torchreid/data/datasets/image/market1501.py    
5- Run the following command after replacing $PATH_TO_DATA with the actual path to the dataset.    

```
python scripts/main.py \
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
--transforms random_flip random_erase \
--root $PATH_TO_DATA

python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root ../market1501 
```
### For Testing    

REPEAT step 4 and step 5

```
python scripts/main.py \
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
--root $PATH_TO_DATA \
  model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
  test.evaluate True \
  test.visrank True 
```


## OSNET on FRIDA Video dataset  
### For Training   
```
python scripts/main.py \     
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \    
  -s FRIDA -t FRIDA \    
  --transforms random_flip random_erase \     
  --root ~/Desktop/ReID/TorchReID/FRIDA \     
  --data-type 'video'   
   
```

### For Testing   
- Testing the FRIDA-trained model on FRIDA
```
python scripts/main.py
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml
 --sources FRIDAimg --targets FRIDAimg
--root ~/Desktop/ReID/TorchReID/FRIDA
model.load_weights ~/Desktop/ReID/TorchReID/deep-person-reid/log/osnet_x1_0_market1501_softmax_cosinelr/model/fridaimg_model.pth.tar-3
test.evaluate True  test.visrank True
```

- Testing the Market1501-trained model on FRIDA   

```
python scripts/main.py
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml
 --sources FRIDAimg --targets FRIDAimg
--root ~/Desktop/ReID/TorchReID/FRIDA
model.load_weights ~/Desktop/ReID/TorchReID/deep-person-reid/log/osnet_x1_0_market1501_softmax_cosinelr/model/market1501_model.pth.tar-250
test.evaluate True  test.visrank True
```
#### Need Help ?      
Refer to: https://kaiyangzhou.github.io/deep-person-reid/user_guide.html
