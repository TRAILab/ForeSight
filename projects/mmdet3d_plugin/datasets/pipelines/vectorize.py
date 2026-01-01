from typing import List, Tuple, Union, Dict

import cv2
import numpy as np
from shapely.geometry import LineString
from numpy.typing import NDArray

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(force=True)
class VectorizeMap(object):
    """Generate vectoized map and put into `semantic_mask` key.
    Concretely, shapely geometry objects are converted into sample points (ndarray).
    We use args `sample_num`, `sample_dist`, `simplify` to specify sampling method.

    Args:
        roi_size (tuple or list): bev range .
        normalize (bool): whether to normalize points to range (0, 1).
        coords_dim (int): dimension of point coordinates.
        simplify (bool): whether to use simpily function. If true, `sample_num` \
            and `sample_dist` will be ignored.
        sample_num (int): number of points to interpolate from a polyline. Set to -1 to ignore.
        sample_dist (float): interpolate distance. Set to -1 to ignore.
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 normalize: bool,
                 coords_dim: int=2,
                 simplify: bool=False, 
                 sample_num: int=-1, 
                 sample_dist: float=-1, 
                 permute: bool=False
        ):
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        self.roi_size = np.array(roi_size)
        self.normalize = normalize
        self.simplify = simplify
        self.permute = permute

        if sample_dist > 0:
            assert sample_num < 0 and not simplify
            self.sample_fn = self.interp_fixed_dist
        elif sample_num > 0:
            assert sample_dist < 0 and not simplify
            self.sample_fn = self.interp_fixed_num
        else:
            assert simplify

    def interp_fixed_num(self, line: LineString) -> NDArray:
        ''' Interpolate a line to fixed number of points.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = np.linspace(0, line.length, self.sample_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()

        return sampled_points

    def interp_fixed_dist(self, line: LineString) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points
    
    def get_vectorized_lines(self, map_geoms: Dict) -> Dict:
        ''' Vectorize map elements. Iterate over the input dict and apply the 
        specified sample funcion.
        
        Args:
            line (LineString): line
        
        Returns:
            vectors (array): dict of vectorized map elements.
        '''

        vectors = {}
        for label, geom_list in map_geoms.items():
            vectors[label] = []
            for geom in geom_list:
                if geom.geom_type == 'LineString':
                    if self.simplify:
                        line = geom.simplify(0.2, preserve_topology=True)
                        line = np.array(line.coords)
                    else:
                        line = self.sample_fn(geom)
                    line = line[:, :self.coords_dim]

                    if self.normalize:
                        line = self.normalize_line(line)
                    if self.permute:
                        line = self.permute_line(line)
                    vectors[label].append(line)

                elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors
    
    def normalize_line(self, line: NDArray) -> NDArray:
        ''' Convert points to range (0, 1).
        
        Args:
            line (LineString): line
        
        Returns:
            normalized (array): normalized points.
        '''

        origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])

        line[:, :2] = line[:, :2] - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        line[:, :2] = line[:, :2] / (self.roi_size + eps)

        return line
    
    def permute_line(self, line: np.ndarray, padding=1e5):
        '''
        (num_pts, 2) -> (num_permute, num_pts, 2)
        where num_permute = 2 * (num_pts - 1)
        '''
        is_closed = np.allclose(line[0], line[-1], atol=1e-3)
        num_points = len(line)
        permute_num = num_points - 1
        permute_lines_list = []
        if is_closed:
            pts_to_permute = line[:-1, :] # throw away replicate start end pts
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(pts_to_permute, shift_i, axis=0))
            flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(flip_pts_to_permute, shift_i, axis=0))
        else:
            permute_lines_list.append(line)
            permute_lines_list.append(np.flip(line, axis=0))

        permute_lines_array = np.stack(permute_lines_list, axis=0)

        if is_closed:
            tmp = np.zeros((permute_num * 2, num_points, self.coords_dim))
            tmp[:, :-1, :] = permute_lines_array
            tmp[:, -1, :] = permute_lines_array[:, 0, :] # add replicate start end pts
            permute_lines_array = tmp

        else:
            # padding
            padding = np.full([permute_num * 2 - 2, num_points, self.coords_dim], padding)
            permute_lines_array = np.concatenate((permute_lines_array, padding), axis=0)
        
        return permute_lines_array
    
    def __call__(self, input_dict):
        if "map_geoms" not in input_dict:
            return input_dict
        map_geoms = input_dict['map_geoms']
        vectors = self.get_vectorized_lines(map_geoms)

        if self.permute:
            gt_map_labels, gt_map_pts = [], []
            for label, vecs in vectors.items():
                for vec in vecs:
                    gt_map_labels.append(label)
                    gt_map_pts.append(vec)
            input_dict['gt_map_labels'] = np.array(gt_map_labels, dtype=np.int64)
            input_dict['gt_map_pts'] = np.array(gt_map_pts, dtype=np.float32).reshape(-1, 2 * (self.sample_num - 1), self.sample_num, self.coords_dim)
        else:
            input_dict['vectors'] = DC(vectors, stack=False, cpu_only=True)
        
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(simplify={self.simplify}, '
        repr_str += f'sample_num={self.sample_num}), '
        repr_str += f'sample_dist={self.sample_dist}), '
        repr_str += f'roi_size={self.roi_size})'
        repr_str += f'normalize={self.normalize})'
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str


@PIPELINES.register_module(force=True)
class RasterizeDrivableArea(object):
    """Build a binary BEV drivable-area mask from boundary linestrings.

    Reads ``input_dict['map_geoms']['boundary']`` (List[LineString] tracing the
    drivable region edges, possibly cut by the ROI) and produces a binary mask
    in lidar-frame ROI by rasterizing boundaries as walls and flood-filling
    from the ego origin. Output: ``input_dict['gt_drivable_mask']`` shape
    ``(H_pix, W_pix)`` uint8 with 1 = drivable.

    Pixel ``(i, j)`` corresponds to lidar XY:
        x = -roi_size[0]/2 + (j + 0.5) * resolution  (lateral)
        y = -roi_size[1]/2 + (i + 0.5) * resolution  (longitudinal)
    """

    def __init__(self, roi_size=(30, 60), resolution=0.5, wall_thickness=2):
        self.roi_size = tuple(roi_size)
        self.resolution = float(resolution)
        self.W = int(round(self.roi_size[0] / self.resolution))
        self.H = int(round(self.roi_size[1] / self.resolution))
        self.wall_thickness = int(wall_thickness)

    def _xy_to_pixel(self, xy):
        cols = (xy[:, 0] + self.roi_size[0] / 2.0) / self.resolution
        rows = (xy[:, 1] + self.roi_size[1] / 2.0) / self.resolution
        return np.stack([cols, rows], axis=-1).round().astype(np.int32)

    def __call__(self, input_dict):
        if 'map_geoms' not in input_dict:
            return input_dict
        boundaries = input_dict['map_geoms'].get('boundary', [])
        walls = np.zeros((self.H, self.W), dtype=np.uint8)
        for ls in boundaries:
            coords = np.asarray(ls.coords)[:, :2]
            if len(coords) < 2:
                continue
            pts = self._xy_to_pixel(coords)
            cv2.polylines(
                walls, [pts], isClosed=False,
                color=1, thickness=self.wall_thickness,
            )
        # Close the ROI box so flood-fill from origin stays within the patch.
        walls[0, :] = 1
        walls[-1, :] = 1
        walls[:, 0] = 1
        walls[:, -1] = 1

        ox = int(round(self.W / 2))
        oy = int(round(self.H / 2))
        ox = max(0, min(self.W - 1, ox))
        oy = max(0, min(self.H - 1, oy))
        if walls[oy, ox] == 1:
            free = np.argwhere(walls == 0)
            if len(free) == 0:
                input_dict['gt_drivable_mask'] = np.zeros(
                    (self.H, self.W), dtype=np.uint8
                )
                return input_dict
            d = (free[:, 0] - oy) ** 2 + (free[:, 1] - ox) ** 2
            i = int(np.argmin(d))
            oy, ox = int(free[i, 0]), int(free[i, 1])

        ff_img = walls.copy()
        ff_mask = np.zeros((self.H + 2, self.W + 2), dtype=np.uint8)
        cv2.floodFill(ff_img, ff_mask, (ox, oy), 2)
        drivable = (ff_img == 2).astype(np.uint8)
        input_dict['gt_drivable_mask'] = drivable
        return input_dict

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(roi_size={self.roi_size}, "
            f"resolution={self.resolution}, wall_thickness={self.wall_thickness})"
        )