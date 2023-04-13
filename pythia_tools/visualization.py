import torch
import sys
import matplotlib.pyplot as plt

def hist_and_box(data_in, bins=50, title=None, save_filename=None):
  data = data_in.detach().cpu()
  # Create a figure with two subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

  # Plot a histogram of the data
  n, bins, patches = ax1.hist(data, bins=bins)
  ax1.set_xlabel('Value')
  ax1.set_ylabel('Frequency')
  ax1.set_title('Histogram')

  # Plot a boxplot of the data
  ax2.boxplot(data)
  ax2.set_xticklabels(['Data'])
  ax2.set_ylabel('Value')
  ax2.set_title('Boxplot')

  if title is not None:
    plt.suptitle(title)

  if save_filename is not None:
    plt.savefig(save_filename)

  # Show the plot
  plt.show()
  plt.close(fig)