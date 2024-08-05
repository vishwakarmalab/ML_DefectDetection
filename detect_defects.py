#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for using model and detecting defects. Input data files should be saved in 
a folder called 'CellFiles' (or 'data_filepath' should appropriately changed), with each 
file being a as a comma separated text file where each line contains the x and y coordinates,
as well as the orientation angle of each cell in radians. 

x and y coordinates should be scaled such that one length unit corresponds to the 
characteristic length of one cell. 

grid_spacing should then be chosen such that the defect core can be captured
by a window with sides of length ~9*grid_spacing

save_filepath, posdef_path and negdef_path should be changed to desired destinations
of defect coordinate files
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import functions as f
import keras

origin = 'Bottom Left' #Set origin as in either 'Bottom Left' or 'Top Left' of images
grid_space = 0.2                            #Choose spacing of interpolation grid
data_filepath = './CellFiles'               #Location of experimental data files
save_filepath = './DefectFiles'             #Path to where defect folders will be located
posdef_path = save_filepath + '/PosDefects' #Location of +1/2 defect files
negdef_path = save_filepath + '/NegDefects' #Location of -1/2 defect files
angles = True #Whether to save the detected defects orientation along with position 

#Load CNN model in old format as TFSMLayer.
model_filepath ='./SavedModel'
model = keras.layers.TFSMLayer(model_filepath, call_endpoint="serving_default")

#Changing TFSMLayer to a full model.
input_layer = keras.Input(shape = (9,9,1))
outputs = model(input_layer)
model_full = keras.Model(input_layer, outputs)
model_full.summary()

#Load experimental data
files = [file for file in sorted(os.listdir(data_filepath))]

#Detect defects
for i,file in enumerate(files):
    file_w_path = os.path.join(data_filepath, file)
    pos_defs,neg_defs = f.DetectDefects(file_w_path,origin,model_full,grid_space,angles)
    f.SaveDefects(posdef_path,negdef_path,pos_defs,neg_defs,i)
    
    if i%20 == 0:
        print('Detected defects in '+str(i)+' files')