"""
Script for embedding input sequences
"""

import os
import sys
import logging
import argparse
import collections

import pandas as pd
import anndata as ad
import scanpy as sc

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tcr")
)
import featurization as ft
import model_utils
import utils

logging.basicConfig(level=logging.INFO)


def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input file. If column-delimited, assume first column is sequences",
    )
    parser.add_argument("outfile", type=str, help="Output file to write")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["B", "AB"],
        type=str,
        default="B",
        help="Input TRB or TRA/TRB pairs",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="wukevin/tcr-bert",
        help="Path to transformer or huggingface model identifier",
    )
    parser.add_argument(
        "-l", "--layer", type=int, default=-1, help="Transformer layer to use"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        required=False,
        default=None,
        type=int,
        help="GPU to run on. If not given or no GPU available, default to CPU",
    )

    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    # Embed the layers
    embeddings = None
    if args.mode == "B":
        trbs = utils.dedup(
            [trb.split("\t")[0] for trb in utils.read_newline_file(args.infile)]
        )
        trbs = [x for x in trbs if ft.adheres_to_vocab(x)]
        logging.info(f"Read in {len(trbs)} unique valid TCRs from {args.infile}")
        obs_df = pd.DataFrame(trbs, columns=['IR_VDJ_1_junction_aa'])
        embeddings = model_utils.get_transformer_embeddings(
            model_dir=args.transformer,
            seqs=trbs,
            layers=[args.layer],
            method="mean",
            device=args.gpu,
        )
    elif args.mode == "AB":
        trabs = utils.dedup(
            [trab.split("\t")[0] for trab in utils.read_newline_file(args.infile)]
        )
        trabs = [x for x in trabs if ft.adheres_to_vocab(x, vocab=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL)]
        logging.info(f"Read in {len(trabs)} unique valid TCRs from {args.infile}")
        obs_df = pd.DataFrame(trabs, columns=['TRA+TRB'])
        obs_df.str.replace('|','+')
        embeddings = model_utils.get_transformer_embeddings(
            model_dir=args.transformer,
            seqs=trabs,
            layers=[args.layer],
            method="mean",
            device=args.gpu,
        )
    assert embeddings is not None

    # Create an anndata object and export it
    embed_adata = ad.AnnData(embeddings, obs=obs_df)
    embed_adata.write_h5ad(args.outfile, compression='gzip')

if __name__ == "__main__":
    main()
