import argparse

import omegaconf

import src.ml.clustering as cl
import src.processing.DataProcessor as dp


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--args_path", type=str, default=None, help="Path to args file")
    # Data
    parser.add_argument(
        "--humans_folder", type=str, default="data/e13", help="Path to humans folder"
    )
    parser.add_argument(
        "--bots_folder", type=str, default="data/twt", help="Path to bots folder"
    )
    parser.add_argument(
        "--data_train",
        type=str,
        default="data/processed/train.csv",
        help="Path to data_train",
    )
    parser.add_argument(
        "--data_test",
        type=str,
        default="data/processed/test.csv",
        help="Path to data_test",
    )
    parser.add_argument(
        "--data_parsed",
        type=str,
        default="data/processed/data_parsed.csv",
        help="Path to complete dataset",
    )

    # Preprocessing
    parser.add_argument(
        "--preprocess", type=bool, default=False, help="Preprocess data"
    )

    # Clustering
    parser.add_argument("--cluster", type=bool, default=False, help="Cluster data")

    # check if we have a config file
    args = parser.parse_args()
    if args.args_path is not None:
        # load the config file
        args = omegaconf.OmegaConf.load(args.args_path)
        args = omegaconf.OmegaConf.to_container(args, resolve=True)

        # convert to args namespace
        args = argparse.Namespace(**args)

    return args


def main():
    args: argparse.Namespace = get_args()
    if args.preprocess:
        dp.DataProcessor(args)()

    if args.cluster:
        cl.Clustering(args)()


if __name__ == "__main__":
    main()
