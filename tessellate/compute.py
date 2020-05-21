import argparse

import xarray as xr
from scipy.sparse import load_npz
from dask.diagnostics import ProgressBar


def load_dims_and_coords(fname, cxy_name):
    grid = xr.open_dataset(fname)

    split_cxy_names = cxy_name.split(',')
    if len(split_cxy_names) == 2:
        cxy_x = grid[split_cxy_names[0]].squeeze()
        cxy_y = grid[split_cxy_names[1]].squeeze()
        cxy = xr.concat([cxy_x, cxy_y], dim='XY')
        cxy = cxy.transpose(*[*cxy.dims[1:], 'XY'])
    else:
        cxy = grid[cxy_name]

    return cxy.dims[:-1], {dim: cxy.coords[dim].values for dim in cxy.dims[:-1]}


def ufunc_multiply(x, M):
    y = M @ x
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_in',
                        type=str,
                        required=True)
    parser.add_argument('--grid_in_cxy',
                        type=str,
                        default='center_xy',
                        required=False)
    parser.add_argument('--grid_out',
                        type=str,
                        required=True)
    parser.add_argument('--grid_out_cxy',
                        type=str,
                        default='center_xy',
                        required=False)
    parser.add_argument('--matrix',
                        type=str,
                        required=True)
    parser.add_argument('--input',
                        type=str,
                        required=True)
    parser.add_argument('--drop',
                        type=str,
                        nargs='+',
                        default=[])
    parser.add_argument('-o',
                        type=str,
                        required=True)
    args = parser.parse_args()

    M = load_npz(args.matrix)
    ds = xr.open_dataset(args.input).drop(args.drop).squeeze()

    in_dims, in_coords = load_dims_and_coords(args.grid_in, args.grid_in_cxy)
    out_dims, out_coords = load_dims_and_coords(args.grid_out, args.grid_out_cxy)

    out_dims = [f'{name}_out' for name in out_dims]
    out_coords = {f'{name}_out': value for name, value in out_coords.items()}

    ds.coords.update(out_coords)
    ds = ds.stack(igrid=in_dims)
    ds = ds.stack(ogrid=out_dims)

    ds = xr.apply_ufunc(
        ufunc_multiply,
        ds, M,
        input_core_dims=[['igrid'], []],
        output_core_dims=[['ogrid']],
        vectorize=True
    )
    ds = ds.unstack('ogrid')

    delayed_obj = ds.to_netcdf(args['o'], compute=False)
    with ProgressBar():
        delayed_obj.compute()


