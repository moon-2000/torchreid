import torchreid

# Preparing the model for training on a GPU.

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
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------


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
#------------------------------------------------------------------------------------------------------------------------------------------------------------------


model = model.cuda()
"""
    cuda():
        - This function is used to transfer the model's parameters to the GPU
            enabling faster training if a compatible GPU is available.
"""
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------



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


scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)
"""
    build_lr_scheduler():
        - This function sets up a learning rate scheduler to adjust the learning rate during training.
            specifying the optimizer (optimizer), the type of scheduler ("single_step"), and the step size (stepsize=20).
        The "single_step" scheduler means to reduce the learning rate after a fixed number of epochs, which is 20 in this case.
"""
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)
"""
    ImageSoftmaxEngine():
        This function is used to build an engine for handling the training process.

    datamanager: This is the data manager object created earlier, specifying the dataset and its configuration.

    model: This is the person re-identification model created and configured earlier.

    optimizer=optimizer: The optimizer for the training process.

    scheduler=scheduler: The learning rate scheduler.

    label_smooth=True: This parameter indicates whether label smoothing is applied during training. 
        Label smoothing is a regularization technique that prevents the model from becoming overconfident 
            in its predictions by smoothing the ground truth labels.
"""
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Run training and test

engine.run(
    save_dir="log/resnet50",
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)
"""
    run():
        - This function is responsible for executing the training and evaluation loop for the person re-identification model.

    save_dir="log/resnet50": specifies the directory where training logs and model checkpoints will be saved.

    max_epoch=60: sets the maximum number of training epochs. The training loop will run for a maximum of 60 epochs.

    eval_freq=10: specifies how often (in terms of epochs) the model should be evaluated on the test set. In this case, it's set to evaluate every 10 epochs.

    print_freq=10: defines how often (in terms of iterations or steps) the training loss should be printed to the console. This can be helpful for monitoring the training progress. In this case, it's set to print every 10 iterations.

    test_only=False: If set to True, the model will only be tested, and no training will be performed. 
        In this case, it's set to False, indicating that both training and testing will be performed.
"""
