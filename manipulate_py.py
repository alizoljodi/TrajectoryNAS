# Import the module to work with the file system
import os

# Define the path to the config.py file
config_path = "/media/asghar/media/FutureDet-NAS/confge.py"

model='''model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[1, 11],
        ds_layer_strides=[1, 4],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False,
        timesteps=timesteps,
        two_stage=TWO_STAGE,
        reverse=REVERSE,
        sparse=SPARSE,
        dense=DENSE,
        bev_map=BEV_MAP,
        forecast_feature=FORECAST_FEATS,
        classify=CLASSIFY,
        wide_head=WIDE,
    ),
)'''
# Check if the file exists
if os.path.isfile(config_path):
    # Open the file in read mode
    with open(config_path, "r") as file:
        # Read the contents of the file
        config_content = file.read()


    # Manipulate the contents of the file
    config_content = config_content.replace('''model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[1, 10],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False,
        timesteps=timesteps,
        two_stage=TWO_STAGE,
        reverse=REVERSE,
        sparse=SPARSE,
        dense=DENSE,
        bev_map=BEV_MAP,
        forecast_feature=FORECAST_FEATS,
        classify=CLASSIFY,
        wide_head=WIDE,
    ),
    )''', model)
    save_path='./cog.py'
    # Open the file in write mode
    with open(save_path, "w") as file:
        # Write the modified contents back to the file
        file.write(config_content)

    # Print a success message
    print("Config file has been updated successfully!")
else:
    # Print an error message if the file does not exist
    print("Config file does not exist!")
