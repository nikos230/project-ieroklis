from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall


class metrics:
    def __init__(self, device='cpu', num_classes=2, average='none'):
        
        # metric dictionary and intialize metrics
        self.metrics = {'accuracy':  MulticlassAccuracy(num_classes=num_classes, average=average).to(device=device),
                        'f1_score':  MulticlassF1Score(num_classes=num_classes, average=average).to(device=device),
                        'iou':       MulticlassJaccardIndex(num_classes=num_classes, average=average).to(device=device),
                        'precision': MulticlassPrecision(num_classes=num_classes, average=average).to(device=device),
                        'recall':    MulticlassRecall(num_classes=num_classes, average=average).to(device=device)
                        }

        # self.metrics = {'accuracy':  BinaryAccuracy().to(device=device),
        #                 'f1_score':  BinaryF1Score().to(device=device),
        #                 'iou':       BinaryJaccardIndex().to(device=device),
        #                 'precision': BinaryPrecision().to(device=device),
        #                 'recall':    BinaryRecall().to(device=device)
        #                 }

        # keep track of metrics values every interation or every epoch for visualization
        self.metrics_values = {}
        for metric_name, metric in self.metrics.items():
            self.metrics_values[metric_name] = []

        # dictionary to save loss for every iterration or every epoch
        self.loss_values = {}


    def reset(self):
        for metric_name, metric in self.metrics.items():
            metric.reset()


    def compute(self):
        # store calculated metrics into a new dictionary
        metric_current_values = {}

        for metric_name, metric in self.metrics.items():
            metric_value = metric.compute().tolist()
            metric_current_values[metric_name] = metric_value
            
            # add metric values to list for later visualization
            self.metrics_values[metric_name].append(metric_value)

        return metric_current_values


    def update(self, output, label):
        for metric_name, metric in self.metrics.items():
            metric.update(output, label)   


    def save_loss(self, epoch, loss_value):
        self.loss_values[epoch] = loss_value
                    




            
