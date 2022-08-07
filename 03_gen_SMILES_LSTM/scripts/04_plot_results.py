
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_theme("talk", "white")
paired_palette = sns.color_palette("Paired")  


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)


parser.add_argument("--jobdir",
                    type=str,
                    default="./output/run_local/",
                    help="Specifies the job directory for the run to visualize.")
args = parser.parse_args()

if __name__ == '__main__':
    
    jobdir = Path(args.jobdir)
    data   = pd.read_csv(jobdir.joinpath("SmilesTrainer_training.csv"))

    
    epochs          = data["epoch"]
    training_loss   = data["training loss"]
    validation_loss = data["validation loss"]
    fraction_valid  = data["fraction valid"]

    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(epochs, training_loss, label="Training loss", color=paired_palette[0] )
    ax.plot(epochs, validation_loss, label="Validation loss", color=paired_palette[1])
    ax.legend()
    ax.set(xlabel="Epoch", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(f"./analysis/{jobdir.name}_loss.png")

    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(epochs, fraction_valid, color=paired_palette[2] )
    ax.set(xlabel="Epoch", ylabel="Fraction valid")
    fig.tight_layout()
    fig.savefig(f"./analysis/{jobdir.name}_fraction_valid.png")
