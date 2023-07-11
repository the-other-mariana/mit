import matplotlib.pyplot as plt
import numpy as np

class LearningPlot:
    def __init__(self, filename) -> None:
        self.file = open(filename, 'r')
        self.metric_data = {}
        self.time_data = {}

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


def main():
    lp = LearningPlot('training_output_5.txt')
    lp.store_data()

if __name__ == '__main__':
    main()