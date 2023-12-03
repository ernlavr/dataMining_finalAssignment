import wandb
import src.utils.utilities as utils
import src.processing.DataProcessor as dp
import src.ml.clustering as cl

def main():
    args = utils.getArgs()
    dataset = utils.get_data(args.data_path)
    data_processor = dp.DataProcessor(dataset)

    clustering = cl.Clustering(data_processor.data)

if __name__ == '__main__':
    main()