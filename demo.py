
import numpy as np
from quiver_engine import server
from quiver_engine.model_utils import register_hook
from torchvision import models

from searched_models.classification.darts import genotypes
from searched_models.classification.darts.model import NetworkCIFAR


if __name__ == "__main__":
    # model = models.resnet18()
    model = NetworkCIFAR(C=16, num_classes=10, layers=20, auxiliary=False, genotype=genotypes.DARTS_V1)

    hook_list = register_hook(model)
    
    server.launch(model, hook_list, input_folder="./data/Cat", image_size=[32, 32], use_gpu=False)
