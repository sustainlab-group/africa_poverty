import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def setup_ax(fig, pos=(1, 1, 1)):
    '''
    Args
    - fig: matplotlib.figure.Figure
    - pos: 3-tuple of int, axes position (nrows, ncols, index)

    Returns: matplotlib.axes.Axes
    '''
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())

    # draw land (better version of cfeature.LAND)
    # land = cfeature.NaturalEarthFeature(
    #     category='physical', name='land', scale='10m',
    #     edgecolor='face', facecolor=cfeature.COLORS['land'], zorder=-1)
    ax.add_feature(cfeature.LAND)

    # draw borders of countries (better version of cfeature.BORDERS)
    countries = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_0_boundary_lines_land', scale='10m',
        edgecolor='black', facecolor='none')
    ax.add_feature(countries)

    # draw coastline (better version of cfeature.COASTLINE)
    coastline = cfeature.NaturalEarthFeature(
        category='physical', name='coastline', scale='10m',
        edgecolor='black', facecolor='none')
    ax.add_feature(coastline)

    # draw lakes (better version of cfeature.LAKES)
    lakes = cfeature.NaturalEarthFeature(
        category='physical', name='lakes', scale='10m',
        edgecolor='face', facecolor=cfeature.COLORS['water'])
    ax.add_feature(lakes)

    # draw ocean (better version of cfeature.OCEAN)
    ocean = cfeature.NaturalEarthFeature(
        category='physical', name='ocean', scale='50m',
        edgecolor='face', facecolor=cfeature.COLORS['water'], zorder=-1)
    ax.add_feature(ocean)

    # draw rivers (better version of cfeature.RIVERS)
    rivers = cfeature.NaturalEarthFeature(
        category='physical', name='rivers_lake_centerlines', scale='10m',
        edgecolor=cfeature.COLORS['water'], facecolor='none')
    ax.add_feature(rivers)

    # draw borders of states/provinces internal to a country
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines', scale='50m',
        edgecolor='gray', facecolor='none')
    ax.add_feature(states_provinces)

    ax.set_aspect('equal')
    gridliner = ax.gridlines(draw_labels=True)
    gridliner.xlabels_top = False
    gridliner.ylabels_right = False
    return ax


def plot_locs(locs, fig=None, pos=(1, 1, 1), figsize=(15, 15), title=None,
              colors=None, cbar_label=None, show_cbar=True, **scatter_kws):
    '''
    Args
    - locs: np.array, shape [N, 2], each row is [lat, lon]
    - fig: matplotlib.figure.Figure
    - pos: 3-tuple of int, axes position (nrows, ncols, index)
    - figsize: list, [width, height] in inches, only used if fig is None
    - title: str
    - colors: list of int, length N
    - cbar_label: str, label for the colorbar
    - show_cbar: bool, whether to show the colorbar
    - scatter_kws: other arguments for ax.scatter

    Returns: matplotlib.axes.Axes
    '''
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = setup_ax(fig, pos)
    if title is not None:
        ax.set_title(title)

    if 's' not in scatter_kws:
        scatter_kws['s'] = 2
    pc = ax.scatter(locs[:, 1], locs[:, 0], c=colors, **scatter_kws)
    if colors is not None and show_cbar:
        cbar = fig.colorbar(pc, ax=ax, fraction=0.03)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    return ax
