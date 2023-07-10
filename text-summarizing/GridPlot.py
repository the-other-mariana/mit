from matplotlib import pyplot as plt
import numpy as np

class GridPlot:
    def __init__(self, rows, cols, out_folder, size=(13, 6)):
        self.rows = rows
        self.cols = cols
        self.out_folder = out_folder

        self.figure, self.axes = plt.subplots(rows, cols, figsize=size)


    def plot_cell(self, i, j, data, mask_data, mask, cmap, title, xtitle, ytitle, xlabels, ylabels, colorbar=True):
        im = self.axes[i, j].imshow(data, cmap=cmap)
        if mask:
            self.axes[i, j].imshow(mask_data, cmap='cool', alpha=0.3)
        if colorbar:
            plt.colorbar(im)
        self.axes[i, j].set_xticks(range(len(xlabels)), labels=xlabels, rotation=45)
        self.axes[i, j].set_yticks(range(len(ylabels)), labels=ylabels,rotation=45)

        self.axes[i, j].set_xlabel(xtitle)
        self.axes[i, j].set_ylabel(ytitle)
        self.axes[i, j].set_title(title)

    def save_plot(self, filename):
        self.figure.tight_layout()
        self.figure.savefig(self.out_folder+filename, dpi=500)

    

        
            


        

    