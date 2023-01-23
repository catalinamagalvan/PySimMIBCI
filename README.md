# PySimMIBCI

You will find here all the codes and instructions needed to reproduce the experiments performed in "A realistic MI-EEG data augmentation approach for improving deep learning in BCI decoding", by Catalina M. Galván, Rubén D. Spies, Diego H. Milone and Victoria Peterson.

## PySimMIBCI Toolbox

Here, all the functions to generate user-specific MI-EEG data with different characteristics are given. Three notebooks example are included: 

1. Example_generate_data_for_augmentation.ipynb, a notebook in which extraction of periodic and aperiodic parameters from real MI-EEG data is implemented and then data that can be used for data augmentation is generated using these user-specific parameters.
2. Example_generate_data_fatigue.ipynb, a notebook that shows the simulation of MI-EEG data with fatigue effects.
3. Example_generate_data_different_user_capabilities.ipynb, a notebook that illustrates how different user capabilities to control a MI-BCI can be simulated.

## FBCNet Toolbox:

You will find here all the codes for using the FBCNetToolbox models, which have been adapted from their [original implementation](https://github.com/ravikiran-mane/FBCNet) to include the possibility to employ data augmentation strategies. 
4. Example_cross_session_data_augmentation.ipynb
5. Example_cross_validation_simdataset.ipynb
