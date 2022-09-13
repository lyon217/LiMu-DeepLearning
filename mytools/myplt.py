# --coding:utf-8--
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline


def set_figsize(figsize=(3.5, 2.5)):
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None,
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    # plt.gcf(),plt.gca() 拿出当前的图表和坐标轴
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        # and的优先级大于or，多个and有假返回第一个假，全真返回最后一个真，
        # 多个or有一个为真，返回第一个真，全假返回最后一个假
        return hasattr(X, 'ndim') and X.ndim == 1 \
               or \
               isinstance(X, list) and not hasattr(X[0], '__len__')

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
