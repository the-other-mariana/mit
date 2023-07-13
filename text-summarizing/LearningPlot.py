import matplotlib.pyplot as plt
import numpy as np

class LearningPlot:
    def __init__(self, filename, size=(10, 6)) -> None:
        self.file = open(filename, 'r')
        self.metric_data = {}
        self.time_data = {}

        self.figure = plt.figure(figsize=size)
        self.grid = self.figure.add_gridspec(3,3)
        self.ax_metric = self.figure.add_subplot(self.grid[1:, :])
        self.ax_time = self.figure.add_subplot(self.grid[0, :])

        # constants to avoid typos
        self.EPOCH = 'epoch'
        self.LOSS = 'loss'
        self.TIME = 'time'

    def is_float(self, string):
        try:
            float_value = float(string)
            return True
        except ValueError:
            return False
    
    def store_data(self):
        lines = self.file.readlines()
        self.metric_data[self.EPOCH] = []
        self.metric_data[self.LOSS] = []
        self.time_data[self.TIME] = []
        for l in lines:
            line = l.lower()
            data = line.split()
            if self.EPOCH in data and self.LOSS in data:
                idx1 = data.index(self.EPOCH)
                idx2 = data.index(self.LOSS)

                self.metric_data[self.EPOCH].append(int(data[idx1+1]))
                self.metric_data[self.LOSS].append(float(data[idx2+1]))
            elif self.TIME in data:
                print(data)
                for d in data:
                    if self.is_float(d):
                        # ignore natural numbers
                        if d not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                            self.time_data[self.TIME].append(float(d))
            else:
                print('[PROMPT] Ignored line')

    def plot_metrics(self):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        set_epochs = set(self.metric_data[self.EPOCH])
        epochs = list(set_epochs)
        data_per_epoch = [[] for _ in epochs]
        epoch_lines = []

        last_index = 0
        xs = list(np.linspace(0, 1000, len(self.metric_data[self.LOSS]), endpoint=True))
        line_height = max(self.metric_data[self.LOSS])
        line_start = min(self.metric_data[self.LOSS])
        xticks = []

        epoch_lines.append([[min(xs), min(xs)], [line_start-line_start*0.05, line_height+line_height*0.05]])
        xticks.append(min(xs))
        for i in range(len(self.metric_data[self.LOSS])):
            data_per_epoch[self.metric_data[self.EPOCH][i]-1].append(self.metric_data[self.LOSS][i])
            if last_index != (self.metric_data[self.EPOCH][i]-1):
                last_index = self.metric_data[self.EPOCH][i]-1
                epoch_lines.append([[xs[i], xs[i]], [line_start-line_start*0.05, line_height+line_height*0.05]])
                xticks.append(xs[i])
        epoch_lines.append([[max(xs), max(xs)], [line_start-line_start*0.05, line_height+line_height*0.05]])
        mid_xticks = [(xticks[i] + xticks[i+1]) * 0.5 for i in range(len(xticks)-1)]
        mid_xticks.append(abs(xticks[-1] - xticks[-2]) + mid_xticks[-1])

        for e in range(len(epoch_lines)):
            self.ax_metric.plot(epoch_lines[e][0], epoch_lines[e][1], color='red', linestyle='--', linewidth=0.8)
        
        # loss plot
        self.ax_metric.plot(xs, self.metric_data[self.LOSS], color='darkred')
        self.ax_metric.set_xlabel(r'Epochs')
        self.ax_metric.set_ylabel(r'Loss')
        self.ax_metric.set_title(r'Loss Across Epochs')

        xlabels = [f'Epoch {i}' for i in epochs]
        self.ax_metric.set_xticks(mid_xticks)
        self.ax_metric.set_xticklabels(xlabels)

        # time plot
        xs = list(range(len(epochs)))
        self.ax_time.plot(xs, np.array(self.time_data[self.TIME]) / 60.0, color='darkred', marker='o')
        self.ax_time.set_xlabel(r'Epochs')
        self.ax_time.set_ylabel(r'Time (Min)')
        self.ax_time.set_title(r'Time per Epoch')

        self.ax_time.set_xticks([i-1 for i in epochs])
        self.ax_time.set_xticklabels(xlabels)
        ylabels = list(np.linspace(min(np.array(self.time_data[self.TIME]) / 60.0), max(np.array(self.time_data[self.TIME]) / 60.0), 5))
        self.ax_time.set_yticks(ylabels)
        self.ax_time.set_yticklabels([f'{y:.1f}' for y in ylabels])
        

        self.figure.tight_layout()
        plt.savefig(f'learning-plot.png', dpi=500)
        plt.show()

        


def main():
    lp = LearningPlot('training_output_5.txt')
    lp.store_data()
    lp.plot_metrics()

if __name__ == '__main__':
    main()