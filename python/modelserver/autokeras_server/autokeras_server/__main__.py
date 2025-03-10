import argparse
import logging

from .model import AutokerasModel
from .model_repository import AutokerasModelRepository
import tserve
from tserve.errors import ModelMissingError



DEFAULT_MODEL_NAME = "structured_data_regressor"
DEFAULT_LOCAL_MODEL_DIR = "D:\\Project\\AM\\AutoML\\tserve\\autokeras\\autokeras\\structured_data_regressor\\best_model"

parser = argparse.ArgumentParser(parents=[tserve.model_server.parser])
parser.add_argument('--model_dir', required=False, default=DEFAULT_LOCAL_MODEL_DIR,
                    help='A URI pointer to the model binary')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
args, _ = parser.parse_known_args()

if __name__ == '__main__':
    model = AutokerasModel(args.model_name, args.model_dir)
    try:
        model.load()
    except ModelMissingError:
        logging.error(f"fail to locate model file for model {args.model_name} under dir {args.model_dir},"
                      f"trying loading from model repository.")

    tserve.ModelServer(registered_models=AutokerasModelRepository(args.model_dir)).start(
        [model] if model.ready else []
    )