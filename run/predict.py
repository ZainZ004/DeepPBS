import numpy as np
import torch
import pickle
from os.path import join as ospj
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import argparse
import json
import os, sys
from functools import reduce

from torch_geometric.data import DataLoader
from deeppbs.nn.utils import loadDataset
from deeppbs.nn import processBatch
from models.model_v2 import Model
from deeppbs import plotPWM, makeLogo, oneHotToSeq, seqToOneHot
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("data_file",
                help="A list of data files.")
arg_parser.add_argument("outpath",
                help="Directory for output files.")
arg_parser.add_argument("-c", "--config", dest="config_file", required=True,
                help="A file storing configuration options.")

arg_parser.add_argument("-p","--plot_each", dest="plot_each", required=False,
                action='store_true',
                help="A file storing configuration options.")

#

# Load the config file
defaults = {
    "no_random": False,
    "debug": False,
    "tensorboard": True,
    "write": True,
    "write_test_predictions": True,
    "single_gpu": False,
    "balance": "unmasked",
    "shuffle": True,
    "weight_method": "dataset",
    "checkpoint_every": 0,
    "eval_every": 2,
    "best_state_metric": "auroc_ovo",
    "best_state_metric_threshold": 0.0,
    "best_state_metric_dataset": "validation", 
    "best_state_metric_goal": "max",
    "remove_zero_class": False
}
ARGS = arg_parser.parse_args()
with open(ARGS.config_file) as FH:
    C = json.load(FH)
# add any missing default args
for key, val in defaults.items():
    if key not in C:
        C[key] = val
# override any explicit args
for key, val in vars(ARGS).items():
    if val is not None:
        C[key] = val


datafiles = [_.strip() for _ in open(ARGS.data_file).readlines()]

#checkpoints = [l.strip() for l in open("./plot_scripts/txts/ppfconv_prot_s_edgef_8.txt","r").readlines()]

if C['readout'] == "all" and C['condition'] == "prot_shape":
    modelname = "DeepPBS"
elif C['readout'] == "all" and C['condition'] == "prot_shape_ag":
    modelname = "DeepPBSwithDNAseqInfo"
elif C['readout'] == "base" and C['condition'] == "prot":
    modelname = "BaseReadout"
elif C['readout'] == "shape" and C['condition'] == "prot_shape":
    modelname = "ShapeReadout"

script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints = [l.strip() for l in open(script_dir + "/plot_scripts/txts/{}.txt".format(modelname),"r").readlines()]
#print(checkpoints)

scalers=[pickle.load(open(script_dir + "/output/{}/scaler.pkl".format(item),"rb")) for item in checkpoints]

checkpoints = [ospj(script_dir+"/output", item, "Model.best.tar") for item in checkpoints]

DLs = []

for i in range(len(scalers)):
    dataset, transforms, info, datafiles = loadDataset(datafiles, C["nc"], C["labels_key"], C["data_dir"],
        cache_dataset=C.get("cache_dataset", False),
        balance=C["balance"],
        remove_mask=False,
        scale=True,
        scaler=scalers[i],
        pre_transform=None,
        feature_mask=None
        )

    DL = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True) 
    DLs.append([batch for batch in DL])

nF_prot = 13
nF_dna = 14



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models = []

for item in checkpoints:
    model = Model(nF_prot, nF_dna, condition=C['condition'], readout=C['readout'])
    model.load_state_dict(torch.load(item, map_location=device)["model_state_dict"]) 
    model.to(device)
    models.append(model)
outpath = ARGS.outpath
if not os.path.exists(outpath):
    os.makedirs(outpath)
if not os.path.exists(outpath+"/npzs"):
    os.makedirs(outpath+"/npzs")

#jaspar_map = open("/home/raktim/cameron/on_jaspar/map.txt","r").readlines()
#def get_jaspar_pwm(uniprot):
#    for line in jaspar_map:
#        if uniprot in line:
#            l = line.strip().split(",")
#            break
#    return l[0] + "." + l[1]

#jdb  = jaspardb(release="JASPAR2022")
def calculate_figure_size(sequence_length, base_width=5.5, base_height=3.5):
    """Calculate appropriate figure size based on sequence length"""
    if sequence_length <= 25:
        return (base_width, base_height)
    else:
        # Scale width proportionally for variable sequence lengths
        scaled_width = base_width + (sequence_length - 25) * 0.2
        return (scaled_width, base_height)

def get_adaptive_ticks_and_labels(sequence_length, seq_list):
    """Determine optimal tick placement and labels based on sequence length"""
    if sequence_length <= 25:
        # Show all positions for short sequences
        ticks = range(sequence_length)
        labels = seq_list
    elif sequence_length <= 50:
        # Show every 2nd position for medium sequences
        ticks = list(range(0, sequence_length, 2)) + [sequence_length - 1]
        labels = [seq_list[i] for i in ticks]
    elif sequence_length <= 100:
        # Show every 5th position for longer sequences
        ticks = list(range(0, sequence_length, 5)) + [sequence_length - 1]
        labels = [seq_list[i] for i in ticks]
    else:
        # Show every 10th position for very long sequences
        ticks = list(range(0, sequence_length, 10)) + [sequence_length - 1]
        labels = [seq_list[i] for i in ticks]

    return ticks, labels

def plot(output, seq, idx=''):
    #fig, (ax3, ax1, ax2) = plt.subplots(3,1)
    gs_kw = dict(width_ratios=[1, 1.4])#, height_ratios=[1, 2])

    # Calculate dynamic figure size based on sequence length
    figsize = calculate_figure_size(len(seq))

    fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=figsize) #layout="constrained")
    ax2 = axd["right"]
    ax3 = axd["upper left"]
    ax1 = axd["lower left"]

    plotPWM(output.T, ax1, xaxis = True)
    makeLogo(output, ax2)
    tosave.append(output)
    plotPWM(np.array(seqToOneHot(seq)).T, ax3, xaxis=True)
    ax2.set_ylabel("bits")
    ax2.set_xlabel("Positions")
    ax1.set_xlabel("Positions")
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylim(0,2)
    ax2.set_title("Prediction")
    ax1.set_title("Prediction")
    ax3.set_title("Sequence")

    # Use adaptive tick placement instead of showing all positions
    ticks, labels = get_adaptive_ticks_and_labels(len(seq), list(seq))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)

    #ax2.set_xlim(0,25)
    plt.tight_layout()
    plt.savefig(ospj(outpath, "{}_{}.svg".format(datafiles[i],idx)))
    plt.close()
    

#
i = 0
tosave = []
#root = oneHotToSeq(dataset[0].y_pwm0[1:-1].data.cpu().numpy())

for data_idx in tqdm(range(len(DLs[0]))):
    
    batch  = DLs[0][data_idx]
    seq = oneHotToSeq(batch.y_pwm0.data.cpu().numpy())
    outputs = []
    for dl_idx in range(len(DLs)):
        batch = DLs[dl_idx][data_idx]
        batch_data = processBatch(device, batch)
        with torch.no_grad():
            outputs.append(torch.softmax(models[dl_idx](batch_data['batch']), dim=1).data.cpu().numpy())
    
    #output = reduce(lambda x, y:x+y, outputs)

    #idx=25 #length considered
    
    #output = (output/len(models))[:idx, :]
    #if primary:
    #else:
    #output = (output/len(models))[idx:, :]
    for j in range(len(outputs)):
        output = outputs[j]
        idx = output.shape[0]//2
        output = output[:idx, :]
        if ARGS.plot_each:
            plot(output, seq, j)

    output = reduce(lambda x, y:x+y, outputs)
    idx = output.shape[0]//2
    output = ((output/len(models))[:idx, :] + np.flip((output/len(models))[idx:, :]))/2
    #output = (output/len(models))[idx:, :]#[::-1,:]
    #plot(np.flip(output), seq, 'ensemble')
    plot(output, seq, 'ensemble')
    
    np.savez_compressed(ospj(outpath, "npzs/%s_predict.npz" % (datafiles[data_idx]) ),
            P=output, Seq=batch.y_hard0.data.cpu().numpy())
    '''
    if r_idx == 1:
        output = output[l_idx:-1]
        seq = seq[l_idx:-1]
    else:
        output = output[l_idx:]
        seq = seq[l_idx:]
    
    if output.shape[0] != 25:
        continue
    '''
    #print(len(seq), output.shape[0])
    i+=1
#np.save("./figs/npy/2r5z_bsc1_pro.npy", np.array(tosave), allow_pickle=True)
