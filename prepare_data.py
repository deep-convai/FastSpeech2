import argparse
import os
import yaml

from preparation import RAVDESS


def main(args, preprocess_config):
    os.makedirs("./lexicon", exist_ok=True)
    os.makedirs("./preprocessed_data", exist_ok=True)
    os.makedirs("./montreal-forced-aligner", exist_ok=True)

    if "RAVDESS" in preprocess_config["dataset"]:
        if args.extract_nonen:
            RAVDESS.extract_nonen(preprocess_config)
        elif args.extract_lexicon:
            RAVDESS.extract_lexicon(preprocess_config)
        elif args.apply_fixed_text:
            RAVDESS.apply_fixed_text(preprocess_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        '--extract_audio',
        help='convert video into .wav file',
        action='store_true',
    )
    parser.add_argument(
        '--extract_nonkr',
        help='extract non korean charactor',
        action='store_true',
    )
    parser.add_argument(
        '--extract_nonen',
        help='extract non english charactor',
        action='store_true',
    )
    parser.add_argument(
        '--extract_lexicon',
        help='extract lexicon and build grapheme-phoneme dictionary',
        action='store_true',
    )
    parser.add_argument(
        '--apply_fixed_text',
        help='apply fixed text to both raw data and filelist',
        action='store_true',
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )

    main(args, preprocess_config)
