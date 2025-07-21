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

    elif loss_function == 'iou':
        return IoULoss()   

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
        dice = 0.

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
            dice += ((2. * intersection) + self.smooth) / (union + self.smooth)

        # return the dice loss
        #dice_loss = 1 - (dice / num_classes).mean() #(1 - dice.mean()) / num_classes
        dice_loss = 1 - (dice.mean() / num_classes)
        #dice_loss = 1 - (dice / num_classes).mean()

        return dice_loss 







# class DiceLoss(nn.Module): 
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):  
#         pred = torch.sigmoid(pred)

#         if pred.shape != target.shape:  
#             target = target.unsqueeze(1)

#         B = pred.shape[0]

#         pred = pred.view(B, -1)       
#         target = target.view(B, -1)

#         intersection = (pred * target).sum(dim=1)  
#         total = pred.sum(dim=1) + target.sum(dim=1)

#         dice_score = (2. * intersection + self.smooth) / (total + self.smooth)
#         dice_loss = 1 - dice_score

#         return dice_loss.mean()  




class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        preds = F.softmax(preds, dim=1)  

        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        preds = preds.view(preds.size(0), preds.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)

        intersection = (preds * targets_one_hot).sum(dim=2)
        union = (preds + targets_one_hot - preds * targets_one_hot).sum(dim=2)

        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1 - iou  
        return loss.mean()






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
        bce_loss = self.bce(pred, target_one_hot)
        

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
          


