import sys
sys.path.insert(0, '.')

from quiver_engine import server
from quiver_engine.model_utils import register_hook
from searched_models.model_generator import model_builder


if __name__ == "__main__":

    model, input_size = model_builder(model_name='darts')

    hook_list = register_hook(model)
    
    server.launch(model, hook_list, input_folder="./data/Cat", image_size=input_size, use_gpu=False)
