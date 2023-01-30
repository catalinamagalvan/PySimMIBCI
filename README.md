
# PySimMIBCI

You will find here all the codes and instructions needed to reproduce the experiments performed in "A realistic MI-EEG data augmentation approach for improving deep learning in BCI decoding", by Catalina M. Galván, Rubén D. Spies, Diego H. Milone and Victoria Peterson.

![Figure 1 fondo blanco](https://user-images.githubusercontent.com/79813952/214372976-539d3e5c-eec8-4925-b8ac-a3dd3cfecf62.png)

## PySimMIBCI Toolbox

Here, all the functions to generate user-specific MI-EEG data with different characteristics are provided in [src](https://github.com/catalinamagalvan/PySimMIBCI/tree/main/src). 

 In lib/FBCNet you will find all the codes for using the FBCNetToolbox models, which have been adapted from their [original implementation](https://github.com/ravikiran-mane/FBCNet) to include the possibility to employ data augmentation strategies.

Notebooks with detailed examples are included in [notebooks](https://github.com/catalinamagalvan/PySimMIBCI/tree/main/notebooks).

1. [Example_generate_data_for_augmentation](https://colab.research.google.com/github/catalinamagalvan/PySimMIBCI/blob/main/notebooks/Example_generate_data_for_augmentation.ipynb): a notebook in which extraction of periodic and aperiodic parameters from real MI-EEG data is implemented and then data that can be used for data augmentation is generated using these user-specific parameters.
2. [Example_generate_data_fatigue](https://colab.research.google.com/github/catalinamagalvan/PySimMIBCI/blob/main/notebooks/Example_generate_data_fatigue.ipynb): a notebook that shows the simulation of MI-EEG data with fatigue effects.
3. [Example_generate_data_different_user_capabilities](https://colab.research.google.com/github/catalinamagalvan/PySimMIBCI/blob/main/notebooks/Example_generate_data_different_user_capabilities.ipynb): a notebook that illustrates how different user capabilities to control a MI-BCI can be simulated.
4. [Example_cross_session_data_augmentation](https://colab.research.google.com/github/catalinamagalvan/PySimMIBCI/blob/main/notebooks/Example_cross_session_data_augmentation.ipynb): a notebook that shows how simulated MI-EEG data can be employed for data augmentation. FBCNet model is trained without and with data augmentation in a cross-session scenario.
5. [Example_cross_validation_simdataset](https://colab.research.google.com/github/catalinamagalvan/PySimMIBCI/blob/main/notebooks/Example_cross_validation_simdataset.ipynb): a notebook that implements a 10-fold cross-validation scenario with simulated MI-EEG data.





