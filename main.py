import wandb
import src.utils.utilities as utils
import src.processing.DataProcessor as dp

def main():
    args = utils.getArgs()
    dataset = utils.get_data(args.data_path)
    data_processor = dp.DataProcessor(dataset)
    pass

if __name__ == '__main__':
    main()