SMALL_FIGURE_SIZE = (7, 6)
LARGE_FIGURE_SIZE = (13, 6)

FIGURE_SETTINGS = {
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": 14,
    "axes.titleweight": "medium",
    "figure.constrained_layout.use": True,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.figsize": SMALL_FIGURE_SIZE,
    "font.family": "Arial",
    "font.size": 12,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

RESIDUAL_PLOT_SETTINGS = {
    "x": "Frequency",
    "y": "Residual",
    "hue": "Sample Name",
    "style": "Component",
}

DEFAULT_LINEPLOT_SETTINGS = {
    "linewidth": 2.5,
    "marker": "o",
    "markersize": 5,
    "markeredgewidth": 0,
    "errorbar": ("se", 95),
    "err_style": "band",
    "sort": False,
}

DEFAULT_JOINT_DISTRIBUTION_PLOT_SETTINGS = {"marginal_kws": {"common_norm": False}}

COVVIS_ANGLE_STEPS = 1
COVVIS_INTERPOLATION_POINTS = 30

PARAMETER_PLOT_FIGURE_SETTINGS = {
    "figure.constrained_layout.use": False,
    "axes.titlesize": FIGURE_SETTINGS["legend.title_fontsize"],
}

DEFAULT_PARAMETER_PLOT_SETTINGS = {
    "catplot_kws": {
        "col_wrap": 3,
        "sharey": False,
        "showfliers": False,
        "boxprops": {"alpha": 0.9},
        "legend": True,
        "height": 3.5,
        "aspect": 1.2,
    },
    "stripplot_kws": {
        "size": 4,
        "jitter": True,
    },
}

DEFAULT_LINEPLOT_GRID_SETTINGS = {
    "kind": "line",
    "facet_kws": {"sharex": True, "sharey": True},
}

DEFAULT_FIT_LINEPLOT_SETTINGS = {
    "size": "Data Origin", 
    "sizes": {"Measured": DEFAULT_LINEPLOT_SETTINGS["linewidth"], 
                "Fitted": DEFAULT_LINEPLOT_SETTINGS["linewidth"] / 2.5}
}