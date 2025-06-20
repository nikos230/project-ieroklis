import torch
import torch.nn as nn
import torch.nn.functional as F


def select_loss_function(loss_function, class_weights):
    # convert class weights from str to intgers and pass them into a tensor
    # class weights only apply to CrossEntropy Loss and BCE Loss Functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if class_weights != 'None':
        class_weights = class_weights.split(', ')
        class_weights = torch.tensor([int(class_weights[0]), int(class_weights[1])], dtype=torch.float).to(device)
    else:
        class_weights = None    
    
    # select the loss function
    if loss_function == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_function == "bce":
        return nn.BCELoss(weight=class_weights)

    elif loss_function == "bce_with_logits":
        return nn.BCEWithLogitsLoss(weight=class_weights)    

    elif loss_function == "dice":  
        return DiceLoss()

    elif loss_function == "bce_dice":
        return BCEDice()

    else:
        return 0



# taken from https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
class DiceLoss:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth

    def __call__(self, pred, target):
        # apply sigmoid to get probabilities
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        dice = 0

        if target.ndim == 4:
            target = target.squeeze(1)
 

        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        for class_ in range(0, num_classes):
            pred_class = pred[:, class_]
            target_class = target_one_hot[:, class_]

            # calc intersection and union
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))

            # calc the dice coef
            dice += (2. * intersection + self.smooth) / (union + self.smooth)

        # return the dice loss
        dice_loss = 1 - (dice / num_classes).mean() #(1 - dice.mean()) / num_classes

        return dice_loss    





class BCEDice(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0, class_weights=None):
        super(BCEDice, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        if class_weights == None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss(class_weights=class_weights)


    def forward(self, pred, target):
 
        num_classes = pred.shape[1]

        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        pred_soft = F.softmax(pred, dim=1)

        # BCE Loss calculation
        bce_loss = F.binary_cross_entropy(pred_soft, target_one_hot, reduction='mean')
        

        # Dice Loss calculation
        dice = 0.0
        for class_ in range(0, num_classes):
            pred_class = pred_soft[:, class_]
            target_class = target_one_hot[:, class_]
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
            dice_class = (2. * intersection + self.smooth) / (union + self.smooth)
            dice += 1 - dice_class.mean()

        dice_loss = dice / num_classes

        # apply weights to each loss and return the total loss
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
          


