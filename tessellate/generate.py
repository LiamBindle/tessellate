import argparse
from typing import Iterable, Union


import numpy as np
import xarray as xr
import pyproj
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import normalize

from tqdm import tqdm


def load_grid(fname, cxy_name, xy_name, proj_init):
    grid = xr.open_dataset(fname)

    split_cxy_names = cxy_name.split(',')
    if len(split_cxy_names) == 2:
        cxy_x = grid[split_cxy_names[0]].squeeze()
        cxy_y = grid[split_cxy_names[1]].squeeze()
        cxy = xr.concat([cxy_x, cxy_y], dim='XY')
        cxy = cxy.transpose(*[*cxy.dims[1:], 'XY'])
    else:
        cxy = grid[cxy_name]

    split_xy_names = xy_name.split(',')
    if len(split_xy_names) == 2:
        xy_x = grid[split_xy_names[0]].squeeze()
        xy_y = grid[split_xy_names[1]].squeeze()
        xy = xr.concat([xy_x, xy_y], dim='XY')
        xy = xy.transpose(*[*xy.dims[1:], 'XY'])
    else:
        xy = grid[xy_name]
    proj = proj_init
    return cxy, xy, proj


def split_xy(xy):
    assert xy[xy.dims[-1]].size == 2
    return xy[..., 0], xy[..., 1]


def conform_and_flatten(xc: xr.DataArray, yc: xr.DataArray, xe: xr.DataArray, ye: xr.DataArray):
    pts_dim = xe.dims[-1]
    assert xe[pts_dim].size == 4
    assert ye.dims[-1] == pts_dim
    assert ye[pts_dim].size == 4
    assert xc.dims == yc.dims
    xe = xe.rename({pts_dim: 'POLYGON_CORNERS'})
    ye = ye.rename({pts_dim: 'POLYGON_CORNERS'})
    xc = xc.stack({'flat': xc.dims})
    yc = yc.stack({'flat': yc.dims})
    xe = xe.stack({'flat': xe.dims[:-1]}).transpose('flat', 'POLYGON_CORNERS')
    ye = ye.stack({'flat': ye.dims[:-1]}).transpose('flat', 'POLYGON_CORNERS')
    return xc, yc, xe, ye


def xeye_to_polygons(xe, ye, in_proj, out_proj):
    if isinstance(xe, float):
        xe = np.array([xe])
        ye = np.array([ye])
    if len(xe) == 0:
        return np.array([])
    xe, ye = pyproj.Transformer.from_crs(crs_from=in_proj, crs_to=out_proj).transform(xe, ye)
    xy = np.moveaxis([xe, ye], 0, -1)
    polygons = np.array([Polygon(outline) for outline in xy])
    return polygons


def xcyc_to_points(xc, yc, in_proj, out_proj):
    if isinstance(xc, float):
        xc = np.array([xc])
        yc = np.array([yc])
    xc, yc = pyproj.Transformer.from_crs(crs_from=in_proj, crs_to=out_proj).transform(xc, yc)
    xy = np.moveaxis([xc, yc], 0, -1)
    polygons = np.array([Point(x, y) for x, y in xy], dtype=object)
    return polygons


def keep_in_extent(xc, yc, xe, ye, proj, extent, proj_extent, ea_proj, chunk_start):
    # Check if input grids fall inside extent
    centers = xcyc_to_points(xc, yc, proj, proj_extent)
    inside = np.array([extent.contains(box_center) for box_center in centers])
    inside_indexes = np.argwhere(inside)[:, 0] + chunk_start

    boxes = xeye_to_polygons(xe[inside], ye[inside], proj, ea_proj)

    valid_boxes = [p.is_valid for p in boxes]

    inside_indexes = inside_indexes[valid_boxes]
    boxes = boxes[valid_boxes]

    index_lut = {id(obox): idx for idx, obox in zip(inside_indexes, boxes)}

    return inside_indexes, boxes, index_lut


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_in',
                        type=str,
                        required=True)
    parser.add_argument('--grid_in_cxy',
                        type=str,
                        default='center_xy',
                        required=False)
    parser.add_argument('--grid_in_xy',
                        type=str,
                        default='outline_xy',
                        required=False)
    parser.add_argument('--grid_in_proj',
                        type=str,
                        default='epsg:4326',
                        required=False)
    parser.add_argument('--grid_out',
                        type=str,
                        required=True)
    parser.add_argument('--grid_out_cxy',
                        type=str,
                        default='center_xy',
                        required=False)
    parser.add_argument('--grid_out_xy',
                        type=str,
                        default='outline_xy',
                        required=False)
    parser.add_argument('--grid_out_proj',
                        type=str,
                        default='epsg:4326',
                        required=False)
    parser.add_argument('--extent',
                        type=float,
                        nargs=4,
                        default=None,
                        required=False)
    parser.add_argument('--extent_proj',
                        type=str,
                        default='epsg:4326',
                        required=False)
    parser.add_argument('--ea_proj',
                        type=str,
                        default='epsg:2163',
                        required=False)
    parser.add_argument('-o',
                        type=str,
                        required=True)
    args = parser.parse_args()

    cxy_in, xy_in, proj_in = load_grid(
        args.grid_in,
        args.grid_in_cxy,
        args.grid_in_xy,
        args.grid_in_proj,
    )
    xc_in, yc_in = split_xy(cxy_in)
    xe_in, ye_in = split_xy(xy_in)
    del cxy_in
    del xy_in

    cxy_out, xy_out, proj_out = load_grid(
        args.grid_out,
        args.grid_out_cxy,
        args.grid_out_xy,
        args.grid_out_proj,
    )
    xc_out, yc_out = split_xy(cxy_out)
    xe_out, ye_out = split_xy(xy_out)
    del cxy_out
    del xy_out

    xc_in_total, yc_in_total, xe_in_total, ye_in_total = conform_and_flatten(xc_in, yc_in, xe_in, ye_in)
    xc_out_total, yc_out_total, xe_out_total, ye_out_total = conform_and_flatten(xc_out, yc_out, xe_out, ye_out)
    del xc_in, yc_in, xe_in, ye_in
    del xc_out, yc_out, xe_out, ye_out

    chunksize = 100000

    size_in = xc_in_total['flat'].size
    size_out = xc_out_total['flat'].size

    xmin, xmax, ymin, ymax = args.extent
    extent = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    proj_extent = args.extent_proj

    ea_proj = args.ea_proj

    M = []

    def weighting_function(ogrid_box, igrid_box):
        return ogrid_box.intersection(igrid_box).area/ogrid_box.area

    for i in tqdm(range(np.ceil(size_out/chunksize).astype(int)), desc='output grid chunk'):
        out_chunk_start = i * chunksize
        out_chunk_end = min(size_out, (i+1)*chunksize)

        xc_out = xc_out_total.isel(flat=range(out_chunk_start, out_chunk_end)).values
        yc_out = yc_out_total.isel(flat=range(out_chunk_start, out_chunk_end)).values
        xe_out = xe_out_total.isel(flat=range(out_chunk_start, out_chunk_end)).values
        ye_out = ye_out_total.isel(flat=range(out_chunk_start, out_chunk_end)).values

        _, out_boxes, out_indexes = keep_in_extent(
            xc_out, yc_out, xe_out, ye_out, proj_out, extent, proj_extent, ea_proj, out_chunk_start
        )

        if len(out_boxes) == 0:
            continue

        rtree = STRtree(out_boxes)

        for j in tqdm(range(np.ceil(size_in/chunksize).astype(int)), desc='input grid chunk'):
            in_chunk_start = j * chunksize
            in_chunk_end = min(size_in-1, (j+1)*chunksize)

            xc_in = xc_in_total.isel(flat=range(in_chunk_start, in_chunk_end)).values
            yc_in = yc_in_total.isel(flat=range(in_chunk_start, in_chunk_end)).values
            xe_in = xe_in_total.isel(flat=range(in_chunk_start, in_chunk_end)).values
            ye_in = ye_in_total.isel(flat=range(in_chunk_start, in_chunk_end)).values

            in_indexes, in_boxes, _ = keep_in_extent(
                xc_in, yc_in, xe_in, ye_in, proj_in, extent, proj_extent, ea_proj, in_chunk_start
            )

            if len(in_boxes) == 0:
                continue

            for ibox_idx, ibox in tqdm(zip(in_indexes, in_boxes), desc='intersect', total=len(in_boxes)):
                entries = [(weighting_function(obox, ibox), (out_indexes[id(obox)], ibox_idx)) for obox in rtree.query(ibox) if obox.intersection(ibox).area > 0]
                M.extend(entries)

    M_dat = [dat for dat, ij in M]
    M_i = [ij[0] for dat, ij in M]
    M_j = [ij[1] for dat, ij in M]
    M = csr_matrix((M_dat, (M_i, M_j)), shape=(size_out, size_in))
    M = normalize(M, norm='l1', axis=1)

    save_npz(args.o, M)

