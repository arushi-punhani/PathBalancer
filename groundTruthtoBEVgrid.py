import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes


# Define your Class Map
CLASS_MAP = {
    'vehicle.car': 1,
    'vehicle.truck': 1,
    'vehicle.bus.bendy': 1,
    'vehicle.bus.rigid': 1,
    'human.pedestrian.adult': 2,
    'human.pedestrian.child': 2,
    'movable_object.barrier': 3
}

def get_bev_gt(nusc, sample_token, grid_res=0.5, grid_size=100,):
    """
    Generates a (200, 200) integer grid where each pixel is a Class ID.
    """
    sample = nusc.get('sample', sample_token)
    pixel_count = int(grid_size / grid_res)
    gt_grid = np.zeros((pixel_count, pixel_count), dtype=np.uint8)
    
    # Use LIDAR_TOP as the coordinate reference
    sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    _, boxes, _ = nusc.get_sample_data(sd_record['token'])
    
    for box in boxes:
        # Determine the class ID from our map
        class_id = 0
        for key, value in CLASS_MAP.items():
            if key in box.name:
                class_id = value
                break
        
        if class_id > 0:
            # Get 4 bottom corners in LiDAR frame
            corners = box.bottom_corners()[:2, :]
            
            # Transform to pixel coordinates (must match input script exactly)
            x_pixels = (-corners[1, :] / grid_res) + (pixel_count / 2)
            y_pixels = (-corners[0, :] / grid_res) + (pixel_count / 2)
            
            polygon = np.stack((x_pixels, y_pixels), axis=1).astype(np.int32)
            
            # fillPoly uses the class_id as the "color"
            cv2.fillPoly(gt_grid, [polygon], int(class_id))
            
    return gt_grid