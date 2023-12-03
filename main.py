import wandb
import src.utils.utilities as utils
import src.processing.DataProcessor as dp
import src.ml.clustering as cl

def main():
    args = utils.getArgs()
    data_processor = dp.DataProcessor(args)

if __name__ == '__main__':
    main()