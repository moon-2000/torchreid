# torchreid

## OSNET model on Market1501   

1- Download the Market1501 dataset from https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html   
2- Make sure to get the structure of the dataset as:   
    ```
    market1501/
        Market-1501-v15.09.15/
            query/
            bounding_box_train/
            bounding_box_test/
    ```
   
3- Clone deep-person-reid repo from https://github.com/KaiyangZhou/deep-person-reid.git   
4- Run the following command after replacing $PATH_TO_DATA with the actual path to the dataset.    

```
python scripts/main.py \
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
--transforms random_flip random_erase \
--root $PATH_TO_DATA
```
