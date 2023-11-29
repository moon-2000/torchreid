# torchreid

## OSNET model on Market1501   
```
python scripts/main.py \
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
--transforms random_flip random_erase \
--root $PATH_TO_DATA
```
