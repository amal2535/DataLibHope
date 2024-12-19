import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')  

class Plotting:
    @staticmethod
    def plot_histogram(data, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency"):
        """Plot a histogram."""
        plt.hist(data, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_scatter(x, y, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
        """Plot a scatter plot."""
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    @staticmethod
    def plot_bar(data, title="Bar Chart", xlabel="X-axis", ylabel="Y-axis"):
        """
        Plot a bar chart.
        :param data: Data for the bar chart (usually a pandas Series or DataFrame).
        :param title: Title of the chart.
        :param xlabel: Label for the X-axis.
        :param ylabel: Label for the Y-axis.
        """
        data.plot(kind='bar')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_correlation_matrix(data):
        """
        Plot the correlation matrix as a heatmap.
        :param data: DataFrame containing the data to plot.
        """
        correlation = data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()