"""
Using link -> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=YjNHjVMOyYlH
For this tutorial we will be fine tuning a pretrained Mask R-CNN model in the 
Penn-Fudan Database for Pedestrian Detection and Segmentation.
It contains 170 Images with 345 instances of Pedestrians.
! We will train an Instance segmentation model on a custom dataset.
"""

import os
from re import L
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned.
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load the images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        # ! Note that we have not converted the mask to RGB since each color corresponds to a different instance.
        # ! with 0 being the background.
        mask = Image.open(mask_path)
        # * Convert the PIL Image to a NumPy Array
        mask = np.array(mask)
        # * instances are encoded as different colors
        obj_ids = np.unique(mask)
        # ! First ID is the background, so remove it.
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get the bounding box co-ordinates for each mask
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            boxes.append([xmin, xmax, ymin, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

    def __len__(self):
        return len(self.imgs)


# Now we load the dataset
dataset = PennFudanDataset('PennFudanPed/')
dataset[0]



# mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
# # Each mask Instance has a different color, from zero to N. where N is the number of instances.
# # In order to make visualization easier, let's add a color palette to the mask.
# mask.putpalette([
#     0, 0, 0, # black background
#     255, 0, 0, # Index 1 is red
#     255, 255, 0, # Index 2 is yellow
#     255, 153, 0, # Index 3 is orange
# ])


# mask.show()

# ! ___________________________________________
# ! Defining Your Model
# * In this tutorial we will be using Mask R-CNN which is based on top of Faster R-CNN. Faster R-CNN is a model that predicts both bounding boxes
# * and class scores for potential objects in the image.
# * Mask R-CNN adds an extra branch into Faster R-CNN, which also predicts segmentations masks for each instance.
# ? There are two ways we can modify a model. 1 - Finetune the Last Layer, 2 - Replace backbone of the Model with a different one.
# %%
# ! 1 -- Finetuning a Pre-trained Model.
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pretrained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# ? Replace the classifier with a new one, that has num_classes which is user-defined.
num_classes = 2
# ? Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# ! Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# %%
# ! -- Modifying the Model to add a different backbone
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pretrained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# FasterCNN needs to know the number of output channels in a backbone.
# For mobilenet_v2 it's 1280 so we need to add it here.
backbone.out_channels = 1280

# ? Let's make the RPN Generate 5 x 3 anchors per spatial location. With 5 different sizes and 3 different Aspect Ratios
# ! RPN --> Region Proposal Network..       AnchorGenerator -> Module that generates the anchors for a set of feature maps.
# ? We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios.
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),        # Anchored Areas to specify levels of areas to perform Detection/Prediction upon.
                                    aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after the rescaling.
# If your backbone returns a Tensor, featmap_names is expected to be [0]. More generally the backbone should return and OrderedDict[Tensor], and in
# featmap_names we can choose which feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# Put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone=backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler
                   )

# %%
# ! In our case given that our dataset is rather small. We will be using the first approach
# ! Hence we will be finetuning the model.
# ! Here we also want to compute the instance segmentation masks, so we will also be using Mask R-CNN.
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")       # ? FPN -> Feature Pyramid Network

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # ! Replace the pretrained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # ! And replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# * This will make the model ready to be trained and evaluated on your custom dataset.
# ! Training and evaluation functions
# In references/detection/, we have a number of helper functions to simplify 
# training and evaluating detection models. Here, we will use references/detection/engine.py, references/detection/utils.py and references/detection/transforms.py.
# Let's copy those files (and their dependencies) in here so that they are available in the notebook
# %%shell

# # Download TorchVision repo to use some files from
# # references/detection
# git clone https://github.com/pytorch/vision.git
# cd vision
# git checkout v0.8.2

# cp references/detection/utils.py ../
# cp references/detection/transforms.py ../
# cp references/detection/coco_eval.py ../
# cp references/detection/engine.py ../
# cp references/detection/coco_utils.py ../

# * Let's write some helper functions for data augmentation / transformation
# * which leverages the functions in reference/detection

from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_transform(train):
    transforms = []
    # Converts an image which is a PIL Image into a PyTorch tensor
    transforms.append(T.ToTensor())
    if train:
        # during the training, randomly flip the training images.
        # and ground truth for data augmentation.
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# %%
# ! Testing FORWARD method
# * Before iterating over the dataset, it is good to see what the model expects during training and inference time on sample data.

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = PennFudanDataset('PenFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True,
    num_workers=4, collate_fn=utils.collate_fn
)

# ! Now for the training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k:v for k, v in t.items()} for t in targets]
output = model(images, targets)         # Returns losses and Detections

# for inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)                  # return the predictions.

# ! Normalization and Standard Deviation is done internally by Mask RCNN
# %%
# ! Putting everything Together
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# split the dataset into train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()         # returns random permutations between given limits

dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn = utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn = utils.collate_fn
)

# %%
# ! Now let's instantiate the model and the optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# ? Add a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ! Now let's train the model for 10 epochs, evaluating at the end of every epoch
from torch.optim.lr_scheduler import StepLR
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, print every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device = device)

# %%
# ! Now that the training is complete, let's see how it works
# * Let's pick one image from the dataset
img, _ = dataset_test[0]
# ? Put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# ! Convert the image, which has been rescaled to 0-1 and had the channels flipped so that we have it in [C, H, W] format
# ? [C, H, W] --> Channel, Height, Width, Each channel is stored as a Column-Major matrix (height, width) of float[numChannels]
# ? Each sample is stored as a column-major matrix (height, width) of float[numChannels] (r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11).
# ? CHW:     RR...R,  GG..G,  BB..B
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

# ! And now let's visualize the top predicted segmentation mask
# ? The masks are predicted as [N, 1, H, W], where N is the number of predictions and are probability maps between 0-1
Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())



# ! So what did we learn in this tutorial?
# ! Created own Pipeline for instance segmentations models on a custom dataset
# ! Leveraged Mask R-CNN model pretrained on COCO train2017 in order to perform transfer learning on this new dataset

# * For a more complete example, which includes multi-machine / multi-gpu training, check references/detection/train.py