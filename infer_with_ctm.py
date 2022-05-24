import random
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np

from main import CombinedTM, prepare_dataset

def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def retrieve_estimates(
    model_dir,
    eval_data_fpath=None,
    n_samples=20,
    **kwargs,
):
    """
    Loads the contextualized topic model then instantiates the encoder portion
    and does a forward pass to get the training-set document-topic estimates

    If `eval_data` is provided, will infer new document-topic estimates for the data
    and the topic-word estimates will __not__ be returned
    """
    model_dir = Path(model_dir)
    config = load_yaml(model_dir / "config.yml")
    train_mode = eval_data_fpath is None

    if train_mode:
        topic_word = np.load(model_dir / "beta.npy")
        if config["n_samples"] == n_samples: # this was created during training
            doc_topic = np.load(model_dir / "train.theta.npy")
            return topic_word, doc_topic
        else:
            eval_data_fpath = Path(config["input_dir"], config["dtm_path"])
    

    eval_data_fpath = Path(eval_data_fpath)
    eval_data_dir = Path(eval_data_fpath).parent
    prefix = Path(eval_data_fpath).name.split(".")[0]
    eval_text_path = eval_data_dir / f"{prefix}.metadata.jsonl"
    vocab_path = Path(config["input_dir"], config["vocab_path"])
    eval_embeds_path = Path(config["input_dir"], "embeddings", config['embeddings_model'], f"{prefix}.embeds.npy")

    eval_dataset = prepare_dataset(
        model_path=config["embeddings_model"],
        dtm_path=eval_data_fpath,
        vocab_path=vocab_path,
        text_path=eval_text_path,
        embeds_path=eval_embeds_path,
        jsonl_text_key=config.get("jsonl_text_key", None), # will guess it
    )

    # re-initialize the model
    ctm = CombinedTM(
        bow_size=eval_dataset.X_bow.shape[1],
        contextual_size=config["contextual_size"],
        n_components=config["num_topics"],
        model_type="prodLDA",
        hidden_sizes=(config["hidden_size"], config["hidden_size"]),
        activation="softplus",
        dropout=config["dropout"],
        learn_priors=not config["no_learn_priors"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        momentum=config["momentum"],
        solver=config["solver"],
        num_epochs=config["num_epochs"],
        reduce_on_plateau=config["reduce_on_plateau"],
        loss_weights=None,
        sample_seed=config["seed"],
    )

    # load the model
    ctm.load(model_dir / "model", epoch=config["num_epochs"] - 1)

    # do the forward pass (if `seed` is None, will use `self.sample_seed`)
    doc_topic = ctm.get_doc_topic_distribution(eval_dataset, n_samples=20)
    return doc_topic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--output_fpath")
    parser.add_argument(
        "--inference_data_file",
        help=(
            "File on which to perform inference, where the filename starts with a <prefix> "
            " followed by a period, e.g., `test.dtm.npz`. Assumes that it has siblings "
            "in the same parent directory: "
            "<prefix>.metadata.jsonl containing text "
            "and embeddings/<embeddings_model>/<prefix>.embeds.npy with embeddings."
        )
    )
    parser.add_argument("--n_samples", default=20, type=int)
    args = parser.parse_args()

    # set the seeds for reproducibility
    doc_topic = retrieve_estimates(
        args.model_dir,
        eval_data_fpath=args.inference_data_file,
        n_samples=args.n_samples,
    )
    Path(args.output_fpath).parent.mkdir(exist_ok=True, parents=True)
    np.save(args.output_fpath, doc_topic)