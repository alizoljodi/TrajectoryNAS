from simanneal import Annealer
import math
from det3d.datasets.nuscenes.eval.detection.constants import getDetectionNames
import sqlite3
import random
import argparse
import json
import os
import sys

import pandas as pd
import os
import argparse
import pynvml
import sys
import pdb
import json
os.environ['MKL_THREADING_LAYER'] = 'GNU'


#tracking_metrics = {"amota" : "AMOTA",
#                    "amotp" : "AMOTP",
#                    "motar" : "MOTAR",
#                    "mota" : "MOTA",
#                    "tp" : "TP",
#                    "fp" : "FP",
#                    "fn" : "FN",
#                    "ids" : "ID_SWITCH",
#                    "frag" : "FRAGMENTED"}

#tracking_dataFrame = { "CLASS" : [],
#                        "AMOTA" : [],
#                        "AMOTP" : [],
#                        "MOTAR" : [],
#                        "MOTA" : [],
#                        "TP" : [],
#                        "FP" : [],
#                        "FN" : [],
#                        "ID_SWITCH" : [],
#                        "FRAGMENTED" : []}

try:
    numDevices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
except:
    pynvml.nvmlInit()
    numDevices = pynvml.nvmlDeviceGetCount()

sys.path.append('/media/asghar/media/FutureDet-NAS')
sys.path.append('/media/asghar/media/FutureDet-NAS/Core/nuscenes-forecast/python-sdk')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='forecast_n3dtf', required=False)
parser.add_argument('--experiment', default='FutureDetection', required=False)
parser.add_argument("--rootDirectory", default="/media/asghar/media/NUSCENES_DATASET_ROOT")
parser.add_argument('--debug', action="store_true")
parser.add_argument("--dataset", default="nusc")
parser.add_argument("--architecture", default="centerpoint")
parser.add_argument("--extractBox", action="store_true")
parser.add_argument("--version", default="v1.0-trainval") #
parser.add_argument("--split", default="val") #
parser.add_argument("--modelCheckPoint", default="latest")
parser.add_argument("--forecast", default=7)
parser.add_argument("--tp_pct", default=0.6)
parser.add_argument("--static_only", action="store_true")
parser.add_argument("--eval_only", action="store_true",default=True)
parser.add_argument("--forecast_mode", default="velocity_forward")
parser.add_argument("--classname", default="car")
parser.add_argument("--rerank", default="last")
parser.add_argument("--cohort_analysis", action="store_true")
parser.add_argument("--jitter", action="store_true")
parser.add_argument("--association_oracle", action="store_true")
parser.add_argument("--postprocess", action="store_true")
parser.add_argument("--nogroup", action="store_true")

parser.add_argument("--K", default=1)
parser.add_argument("--C", default=1)
parser.set_defaults(debug=True)
args = parser.parse_args()

model = args.model

experiment = args.experiment
dataset = args.dataset
architecture = args.architecture
configPath = "{dataset}_{architecture}_{model}_detection.py".format(dataset=dataset,
                                                                    architecture=architecture,
                                                                    model=model)
architecture = args.architecture
experiment = args.experiment
rootDirectory = args.rootDirectory
model = args.model
dataset = args.dataset
version = args.version
split = args.split
extractBox = args.extractBox
modelCheckPoint = args.modelCheckPoint
forecast = args.forecast
forecast_mode = args.forecast_mode
classname = args.classname
rerank = args.rerank
tp_pct = args.tp_pct
static_only = args.static_only
eval_only = args.eval_only
cohort_analysis = args.cohort_analysis
K = args.K
C = args.C
jitter = args.jitter
association_oracle = args.association_oracle
postprocess = args.postprocess
nogroup = args.nogroup
#print(configPath)
#input('fr')
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# numDevices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
try:
    numDevices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
except:
    pynvml.nvmlInit()
    numDevices = pynvml.nvmlDeviceGetCount()

class SimAnneal(Annealer):
    def __int__(self,state,dir,init_model):
        super(SimAnneal,self).__init__(state)
        self.num = 0
        self.last_e = math.inf
        self.past_model = init_model
        self.best = math.inf
        self.path = dir

        self.db = dir + '/models.db'
        print(self.db)

        conn = sqlite3.connect(self.db)

        c = conn.cursor()

        c.execute('''CREATE TABLE bests (num int, conf text,AP real, energy real)''')
        conn.commit()
        c.execute('''CREATE TABLE _all_ (num int, conf text,AP real, energy real)''')
        conn.commit()
        conn.close()
    def move(self):
        if random.choice([True,False]):
            if random.choice([True, False]):
                self.state['layers'][0]=self.state['layers'][0]+random.choice([-1,1])
            else:
                self.state['layers'][1] = self.state['layers'][1] + random.choice([-1, 1])
        else:
            change = random.choice(['reg', 'height', 'dim', 'rot', 'vel'])

            if random.choice([True, False]):
                self.state[change][0]=self.state[change][0]+ random.choice([-1, 1])
            else:
                self.state[change][1] = self.state[change][1] + random.choice([-1, 1])


        config_path = "/media/asghar/media/FutureDet-NAS/confge.py"
        model = '''model = dict(
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
                layer_nums=[{layer_num1}, {layer_num2}],
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
                common_heads={'reg': ({reg_1}, {reg_2}), 'height': ({he_1}, {he_2}), 'dim':({dim_1}, {dim_2}), 'rot':({rot_1}, {rot_2}), 'vel': ({vel_1}, {vel_2})},
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
        )'''.format(layer_num1=state['layers'][0],
                    layer_num2=state['layers'][1],
                    reg_1=state['reg'][0],
                    reg_2=state['reg'][1],
                    he_1=state['height'][0],
                    he_2=state['height'][1],
                    dim_1=state['dim'][0],
                    dim_2=state['dim'][1],
                    rot_1=state['rot'][0],
                    rot_2=state['rot'][1],
                    vel_1=state['vel'][0],
                    vel_2=state['vel'][1])
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
            layer_nums=[1, 1],
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
        self.save_path=os.path.join(self.path,str(self.num))
        os.mkdir(path=self.save_path)
        with open(self.save_path+'config.py','w') as file:
            file.write(config_content)
        return self.energy()
    def energy(self):
        detection_metrics = {"trans_err": "ATE",
                             "scale_err": "ASE",
                             "orient_err": "AOE",
                             "vel_err": "AVE",
                             "attr_err": "AAE",
                             "avg_disp_err": "ADE",
                             "final_disp_err": "FDE",
                             "miss_rate": "MR",
                             # "reverse_avg_disp_err" : "RADE",
                             # "reverse_final_disp_err" : "RFDE",
                             # "reverse_miss_rate" : "RMR",
                             }

        detection_dataFrame = {"CLASS": [],
                               "mAP": [],
                               "mAR": [],
                               "mFAP": [],
                               "mFAR": [],
                               "mAAP": [],
                               "mAAR": [],
                               "ATE": [],
                               "ASE": [],
                               "AOE": [],
                               "AVE": [],
                               "AAE": [],
                               "ADE": [],
                               "FDE": [],
                               "MR": [],
                               "mFAP_MR": [],
                               #                        "RADE" : [],
                               #                        "RFDE" : [],
                               #                        "RMR" : []
                               }
        os.system(
            "python  ./tools/train.py configs/{architecture}/{configPath} --seed 0 --work_dir {det_dir}".format(
                architecture=architecture,
                configPath=configPath,
                det_dir=self.save_path))
        os.system(
            "python ./tools/dist_test.py configs/{architecture}/{configPath} {extractBox} --work_dir {det_dir} --checkpoint {det_dir}/{modelCheckPoint}.pth --modelCheckPoint {modelCheckPoint} --forecast {forecast} --forecast_mode {forecast_mode} --classname {classname} --rerank {rerank} --tp_pct {tp_pct} {static_only} {eval_only} {cohort_analysis} {jitter} {association_oracle} {postprocess} {nogroup} --K {K} --C {C} --split {split} --version {version} --root {rootDirectory}".format(
                architecture=architecture,
                configPath=configPath,
                extractBox="--extractBox" if extractBox else "",
                det_dir=self.save_path,
                modelCheckPoint=modelCheckPoint,
                forecast=forecast,
                forecast_mode=forecast_mode,
                classname=classname,
                rerank=rerank,
                tp_pct=tp_pct,
                K=K,
                C=C,
                eval_only="--eval_only" if eval_only else "",
                static_only="--static_only" if static_only else "",
                cohort_analysis="--cohort_analysis" if cohort_analysis else "",
                jitter="--jitter" if jitter else "",
                association_oracle="--association_oracle" if association_oracle else "",
                postprocess="--postprocess" if postprocess else "",
                nogroup="--nogroup" if nogroup else "",
                split=split,
                version=version,
                rootDirectory=rootDirectory))

        logFile = json.load(open(self.save_path + "/metrics_summary.json"))

        detection_dataFrame["CLASS"] = detection_dataFrame["CLASS"] + getDetectionNames(cohort_analysis)

        for classname in detection_dataFrame["CLASS"]:
            detection_dataFrame["mAP"].append(logFile["mean_dist_aps"][classname])
            detection_dataFrame["mAR"].append(logFile["mean_dist_ars"][classname])

            detection_dataFrame["mFAP"].append(logFile["mean_dist_faps"][classname])
            detection_dataFrame["mFAR"].append(logFile["mean_dist_fars"][classname])

            detection_dataFrame["mAAP"].append(logFile["mean_dist_aaps"][classname])
            detection_dataFrame["mAAR"].append(logFile["mean_dist_aars"][classname])

            detection_dataFrame["mFAP_MR"].append(logFile["mean_dist_faps_mr"][classname])

        classMetrics = logFile["label_tp_errors"]
        for metric in detection_metrics.keys():
            for classname in detection_dataFrame["CLASS"]:
                detection_dataFrame[detection_metrics[metric]].append(classMetrics[classname][metric])

        detection_dataFrame = pd.DataFrame.from_dict(detection_dataFrame)

        if not os.path.isdir("results/" + experiment + "/" + model):
            os.makedirs("results/" + experiment + "/" + model)

        filename = "results/{experiment}/{model}/{dataset}_{architecture}_{model}_{forecast}_{forecast_mode}_{rerank}_tp{tp_pct}_K{K}_{cohort}{static_only}{association_oracle}{jitter}{postprocess}detection_{modelCheckPoint}.csv".format(
            experiment=experiment, model=model, dataset=dataset, architecture=architecture,
            forecast="t{}".format(forecast), forecast_mode=forecast_mode, rerank=rerank, tp_pct=tp_pct, K=K,
            jitter="{}jitter_".format(C) if jitter else "", cohort="cohort_" if cohort_analysis else "",
            static_only="static_" if static_only else "", association_oracle="oracle_" if association_oracle else "",
            postprocess="pp_" if postprocess else "", modelCheckPoint=modelCheckPoint)
        detection_dataFrame.to_csv(filename, index=False)
state={'layers':[1,1],'reg': [2, 2], 'height': [1, 2], 'dim':[3, 2], 'rot':[2, 2], 'vel': [2, 2]}