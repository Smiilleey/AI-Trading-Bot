# core/order_flow_visualizer.py

import matplotlib.pyplot as plt
import numpy as np

class OrderFlowVisualizer:
    def __init__(self):
        self.frames = []

    def add_frame(self, bid_volume, ask_volume):
        """
        Stores a single frame of order flow data.
        Each frame = list of bid and ask volumes at price levels.
        """
        self.frames.append((bid_volume, ask_volume))

    def plot_last_frame(self):
        """
        Plots delta footprint, bid/ask heatmap, and combined intensity.
        """
        if not self.frames:
            print("No order flow data to display.")
            return

        bid, ask = self.frames[-1]
        levels = len(bid)
        prices = np.arange(levels)

        delta = np.array(ask) - np.array(bid)
        total_volume = np.array(ask) + np.array(bid)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Delta Footprint
        axs[0].barh(prices, delta, color=["green" if d > 0 else "red" for d in delta])
        axs[0].set_title("Delta Footprint")
        axs[0].invert_yaxis()

        # Plot 2: Bid/Ask Heatmap
        heatmap = np.vstack([bid, ask])
        axs[1].imshow(heatmap, cmap='coolwarm', aspect='auto')
        axs[1].set_title("Bid/Ask Heatmap")
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels(["Bid", "Ask"])

        # Plot 3: Total Volume Intensity
        axs[2].barh(prices, total_volume, color="blue")
        axs[2].set_title("Volume Intensity")
        axs[2].invert_yaxis()

        plt.tight_layout()
        plt.show()

    def reset(self):
        self.frames = []
