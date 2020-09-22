import torch
from searched_models.segmentation.auto_deeplab.build_autodeeplab import Retrain_Autodeeplab
from config.model_config import auto_deeplab_retrain_args


def model_builder(model_name):
    if model_name == 'resnet18':
        input_size = [250, 250]
        from torchvision import models
        model = models.resnet18()
    elif model_name == 'darts':
        input_size = [32, 32]
        from searched_models.classification.darts import genotypes
        from searched_models.classification.darts.model import NetworkCIFAR
        model = NetworkCIFAR(C=16, num_classes=10, layers=20, auxiliary=False, genotype=genotypes.DARTS_V1)
    elif model_name == 'auto_deeplab':
        input_size = [513, 513]
        args = auto_deeplab_retrain_args()
        args.num_classes = 19
        model = Retrain_Autodeeplab(args)
        model.eval()
    else:
        raise ValueError('No Defined Model!')

    return model, input_size


if __name__ == '__main__':
    model = model_builder(model_name='resnet18')
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
