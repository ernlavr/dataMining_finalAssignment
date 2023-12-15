import wandb
import src.utils.utilities as utils
import src.processing.DataProcessor as dp
import src.ml.dimensionalityReduction as dr
import src.ml.clustering as cl
import src.ml.sentiment_anal as sa

def main():
    args = utils.getArgs()

    if args.preprocess:
        data_processor = dp.DataProcessor(args)

    if args.cluster:
        clustering = cl.Clustering(args)

if __name__ == '__main__':
    main()