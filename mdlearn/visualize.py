
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
import matplotlib.pyplot as plt

from . import metrics


class LinearVisualizer:
    """ Compare 1D data.
    """

    colormap = plt.cm.prism(np.linspace(0, 1, 15))

    def __init__(self, x, y, name=None, label='default'):
        """ x, y: array-like objects.
            x is used for x axis and y is used for y axis;
            name: list of y name / None (use integer as name)
        """
        self.groups = {}
        self.append(x, y, name, label)

    def update_y(self, y, label='default'):
        """ Update y data
        """

        self.groups[label] = (self.groups[label][0], y, self.groups[label][2])

    def append(self, x, y, name=None, label='new'):

        if name is None:
            name = [str(i) for i in range(len(y))]

        assert len(y) > 0
        assert len(y) == len(x)
        assert len(name) == len(y)

        self.groups[label] = (x, y, name)

    def dump(self, filename):

        output = open(filename, 'w')

        for label, g in self.groups.items():
            for i in range(len(g[0])):
                output.write('%s\t%s\t%s\t%s\n' % (label, g[2][i], g[0][i], g[1][i]))

        output.close()

    def dump_bad_molecules(self, filename, threshold=0.1):
        output = open(filename, 'w')

        for x, y, name in self.groups.values():
            for i in np.where(np.abs(y - x) > threshold * np.abs(x))[0]:
                output.write(f'{name[i]}\t{x[i]}\t{y[i]}\n')

        output.close()

    def scatter_yy(self, ref='line', annotate_threshold=0.1, figure_name=None, **kwargs):
        """ Plot scatter(y_ref, y)
            Additional arguments will be passed to plt.scatter
        """
        plt.figure(figure_name)

        if ref is not None:

            yrange = (
                min( [min(d[0]) for d in self.groups.values()] + [min(d[1]) for d in self.groups.values()] ),
                max( [max(d[0]) for d in self.groups.values()] + [max(d[1]) for d in self.groups.values()] )
            )

            if ref == 'line':

                plt.plot(yrange, yrange, '-', color='b', lw=0.4)

        groupnames = list(self.groups.keys())

        plots = [
            plt.scatter(self.groups[l][0], self.groups[l][1], color=LinearVisualizer.colormap[i], **kwargs) for i, l in enumerate(groupnames)
            ]
        plt.legend(plots, groupnames, loc='lower right')

        if annotate_threshold >= 0:

            for x, y, name in self.groups.values():
                for i in np.where(np.abs(y - x) > annotate_threshold * np.abs(x))[0]:
                    plt.annotate(name[i].split()[0], (x[i], y[i]))


    def scatter_error(self, ref='line', error='abs', annotate_threshold=0.1, figure_name=None, **kwargs):

        plt.figure(figure_name)
        groupnames = list(self.groups.keys())

        error_func = metrics.relative_error if error == 'ref' else metrics.absolute_error

        plots = [
            plt.scatter(self.groups[l][0], error_func(self.groups[l][0], self.groups[l][1]), color=LinearVisualizer.colormap[i], **kwargs) for i, l in enumerate(groupnames)
            ]
        plt.legend(plots, groupnames, loc='lower right')

        if ref is not None:

            yrange = (
                min([min(d[0]) for d in self.groups.values()]),
                max([max(d[0]) for d in self.groups.values()]))

            if ref == 'line':

                plt.axhline(0, yrange[0], yrange[1], color='b', lw=0.4)


        if annotate_threshold >= 0:

            for x, y, name in self.groups.values():
                for i in np.where(metrics.relative_error(x, y) > annotate_threshold)[0]:
                    plt.annotate(name[i].split()[0], (x[i], y[i]))


    def scatter_xy(self, label='default', data='relerr', figure_name=None, **kwargs):

        plt.figure(figure_name)
        y1, y2, x = self.groups[label]

        if data == 'relerr':
            y = metrics.relative_error(y1, y2)

        elif data == 'abserr':
            y = metrics.absolute_error(y1, y2)

        elif data == 'predict':
            y = y2

        elif data == 'real':
            y = y1

        else:
            raise RuntimeError()

        plt.scatter(x, y, **kwargs)


    def hist_error(self, label='default', errtype='rel', figure_name=None, **kwargs):
        x, y = self.groups[label][:2]

        if errtype == 'rel':
            err = metrics.abs_relative_error(x, y)

        elif errtype == 'abs':
            err = metrics.abs_absolute_error(x, y)

        else:
            raise RuntimeError()

        plt.figure(figure_name)
        plt.hist(err, **kwargs)
