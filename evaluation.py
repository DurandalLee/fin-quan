import numpy as np
import matplotlib.pyplot as plt
import math


class Evaluation:
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
        self.cos = 0.
        self.pre_accuracy = []
        self.pearson_cor = 0.

    # point multiplication of sequence
    @staticmethod
    def dot_product(vector1: np.array, vector2: np.array):
        return sum(vector1 * vector2)

    # cosine similarity
    def cos_similarity(self):
        xy_dot = self.dot_product(self.x, self.y)
        xx_dot = self.dot_product(self.x, self.x)
        yy_dot = self.dot_product(self.y, self.y)
        self.cos = xy_dot / math.sqrt(xx_dot * yy_dot)

    # percentage of error and accuracy
    def percentage(self):
        for fun_i in range(len(self.x)):
            error = (self.y[fun_i] - self.x[fun_i]) / self.y[fun_i]
            self.pre_accuracy.append(1 - error)

    # calculate the pearson correlation
    def pearson(self):
        self.pearson_cor = np.corrcoef(self.x.flatten(), self.y.flatten())[0, 1]

    def show_evaluate_result(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ticks_list = list(range(len(self.y) + 1))[::20]

        ax1.clear()
        ax1.plot(self.x, "r--", label="predict_data")
        ax1.plot(self.y, "c-", label="true_data")
        ax1.set_xticks(ticks_list)
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.legend(loc="best")
        ax1.grid(linestyle='-.')

        self.cos_similarity()
        self.pearson()
        self.percentage()

        ax2.clear()
        ax2.plot(self.pre_accuracy, "k-", label="accuracy")
        ax2.set_xticks(ticks_list)
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Accuracy")
        ax2.grid(linestyle='-.')

        print("cosine similarity=" + str(self.cos) +
              ",pearson correlation=" + str(self.pearson_cor))
