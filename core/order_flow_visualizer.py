# core/order_flow_visualizer.py

import matplotlib.pyplot as plt
import numpy as np

class OrderFlowVisualizer:
    def __init__(self):
        self.frames = []
        self.frame_tags = []  # Each frame can have symbolic tags (e.g. "absorption", "flip", "session_open")

    def add_frame(self, bid_volume, ask_volume, tags=None):
        """
        Stores a single frame of order flow data, plus any symbolic tags.
        Each frame = (bid list, ask list, [optional tags])
        """
        self.frames.append((bid_volume, ask_volume))
        self.frame_tags.append(tags or [])

    def plot_last_frame(self, with_tags=True):
        """
        Plots delta footprint, bid/ask heatmap, and total volume.
        Overlays symbolic tags if present.
        """
        if not self.frames:
            print("No order flow data to display.")
            return

        bid, ask = self.frames[-1][:2]
        tags = self.frame_tags[-1] if with_tags and self.frame_tags else []
        levels = len(bid)
        prices = np.arange(levels)

        delta = np.array(ask) - np.array(bid)
        total_volume = np.array(ask) + np.array(bid)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Order Flow Visualizer")

        # Plot 1: Delta Footprint
        bars = axs[0].barh(prices, delta, color=["green" if d > 0 else "red" for d in delta])
        axs[0].set_title("Delta Footprint")
        axs[0].invert_yaxis()
        if tags:
            for i, tag in enumerate(tags):
                axs[0].text(0, i, tag, fontsize=8, color='blue')

        # Plot 2: Bid/Ask Heatmap
        heatmap = np.vstack([bid, ask])
        axs[1].imshow(heatmap, cmap='coolwarm', aspect='auto')
        axs[1].set_title("Bid/Ask Heatmap")
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels(["Bid", "Ask"])
        if tags:
            for i, tag in enumerate(tags):
                axs[1].text(len(bid)//2, i, tag, fontsize=8, color='yellow')

        # Plot 3: Total Volume Intensity
        axs[2].barh(prices, total_volume, color="blue")
        axs[2].set_title("Volume Intensity")
        axs[2].invert_yaxis()
        if tags:
            for i, tag in enumerate(tags):
                axs[2].text(0, i, tag, fontsize=8, color='purple')

        plt.tight_layout()
        plt.show()

    def plot_frame(self, idx=-1, with_tags=True):
        """Plots a specific frame by index (default = last)."""
        if not self.frames or abs(idx) > len(self.frames):
            print("Frame index out of range.")
            return
        bid, ask = self.frames[idx][:2]
        tags = self.frame_tags[idx] if with_tags and self.frame_tags else []
        levels = len(bid)
        prices = np.arange(levels)
        delta = np.array(ask) - np.array(bid)
        total_volume = np.array(ask) + np.array(bid)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        bars = axs[0].barh(prices, delta, color=["green" if d > 0 else "red" for d in delta])
        axs[0].set_title("Delta Footprint")
        axs[0].invert_yaxis()
        if tags:
            for i, tag in enumerate(tags):
                axs[0].text(0, i, tag, fontsize=8, color='blue')
        heatmap = np.vstack([bid, ask])
        axs[1].imshow(heatmap, cmap='coolwarm', aspect='auto')
        axs[1].set_title("Bid/Ask Heatmap")
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels(["Bid", "Ask"])
        if tags:
            for i, tag in enumerate(tags):
                axs[1].text(len(bid)//2, i, tag, fontsize=8, color='yellow')
        axs[2].barh(prices, total_volume, color="blue")
        axs[2].set_title("Volume Intensity")
        axs[2].invert_yaxis()
        if tags:
            for i, tag in enumerate(tags):
                axs[2].text(0, i, tag, fontsize=8, color='purple')
        plt.tight_layout()
        plt.show()

    def reset(self):
        self.frames = []
        self.frame_tags = []
