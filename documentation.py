import torchreid


datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="market1501",
    targets="market1501",
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=["random_flip", "random_crop"]
)

"""
    ImageDataManager():
        - This function configures the datamanager with the specified parameters
            the resulting datamanager object can be used to access training and testing data whenever needed. 
    
    root: the root directory specifies where the dataset is located.

    sources: the dataset used for training.
    targets: define the dataset used for testing.

    height and width: set the height and width of the input images. "Images will be resized to these dimensions".

    batch_size_train: specify the batch size for training. During training, the data will be processed in batches of 32 images. 
    batch_size_test: Specify the batch size for testing. During testing, batches of 100 images will be used.

    transforms: specifies the data augmentation techniques applied during training. 
        "random_flip" and "random_crop": randomly flipping and cropping images during training can help improve the model's ability to generalize.
"""
#-----------------------------------------------------------------------------------------------------------------------------------------------------


model = torchreid.models.build_model(
    name="resnet50",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=True
)

"""
    build_model(): 
        - This function is used to build a person re-identification model.

    name="resnet50": specifies the architecture of the model. 
        In this case, it's using the ResNet-50 architecture.

    num_classes=datamanager.num_train_pids: sets the number of output classes or identities for the model. 
        It's set to the number of unique person identities in the training dataset.

    loss="softmax": specifies the loss function used during training. 
        "Softmax" typically refers to the softmax cross-entropy loss, which is commonly used for classification tasks.

    pretrained=True: Indicates whether to use pre-trained weights for the ResNet-50 model. 
        Setting it to True means using weights that have been pre-trained on a large dataset.
"""
#-------------------------------------------------------------------------------------------------------------------------------------------------------------


model = model.cuda()
"""
    cuda():
        - This function is used to transfer the model's parameters to the GPU
            enabling faster training if a compatible GPU is available.
"""
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)
"""
    build_optimizer():
        - This function creates an optimizer for training the model
            specifying the model (model), the optimization algorithm ("adam"), and the learning rate (lr=0.0003). 
            The Adam optimizer is a popular optimization algorithm used for training deep neural networks.
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

