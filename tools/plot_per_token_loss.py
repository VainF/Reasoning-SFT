import json
import matplotlib.pyplot as plt

def plot_first_10_samples(json_file_path):
    """
    Load the first 10 samples from the given JSON file,
    extract 'per_token_loss', and plot them as line charts.
    """
    first_10_token_losses = []

    # Read the file and collect up to 10 samples
    with open(json_file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            data = json.loads(line)
            if "per_token_loss" in data:
                first_10_token_losses.append(data["per_token_loss"])

    # Plot each sample in one figure (multiple lines)
    plt.figure(figsize=(12, 6))
    for i, losses in enumerate(first_10_token_losses, start=1):
        # Create an x range for each token
        x = range(len(losses))  
        # Plot as a line: the x-values are token indices, y-values are the losses
        plt.plot(x, losses, label=f"Sample {i}")
    
    # y [0-20]
    plt.ylim(0, 20)
    plt.title("Per-token Loss for First 10 Samples")
    plt.xlabel("Token Index")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/first_10_samples_loss.png")


# Example usage – call this AFTER your main data-processing script finishes
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose your main script saved results to:
    output_file_path = "assets/data/generated/open_thoughts_r1_7b_greedy-loss.json"

    # Simply call the plot function using the same path
    plot_first_10_samples(output_file_path)
