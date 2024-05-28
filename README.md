# Empatia_DL
This project is for the Python code for Empatia with DL methods. Using WEMAC and WESAD data base.

The paper related to this project has been published in _DOI: 10.1109/JBHI.2024.3392373_

Readme for Empatia deep learning

**Please, pay attention to all of the file locations in the project. Change them to yours. That is the most impotant thing.**

Environment
GPU: Nvidia A30
Cuda: 12.2
Driver: 535.154.05
Torch: 2.0.0+cu118


WEMAC
1.BVP_Signal_Features, GSR_Signal_Features and SKT_Signal_Features.py are the python files for calculating all of the 123 physiological features. 
2.Pack_all_data.py used for generating a json file to save those 123 features because it would be a huge time consumer if we call the WEMAC matlab data set every time. Remember change the location of .mat file and label file. If you want to use IT06.mat, please use Transfer_label.py firstly.
3.Data_normalization.py created for feature normalization because the calculation functions have different data level. We use FWN to normalize them and generate a .log for recording which normalization function should be used for a specific feature. Please change the name of .log.
4.Create_feature_maps.py includes the method how to generate 2D feature maps based on the json file and FWN log file.
5.Then we could start to train those feature maps using different DL models. In General_F1.py, we tested the training process of train:test = 8:2 and evaluate the performance according to acc and f1. In Laso_F1.py we test the data split strategy of LASO for the comparison with Bindi. The last one is train_rfe_fisher.py which is using RFE feature selection and fisher score to improve performance.

WESAD
1.The feature calculation methods are just same as WEMAC.
2.Feature_extraction_wesad.py record the methods for writing all of the features into several json files. Please check the root. You can download WESAD here https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
3.Data_normalization.py is just similar to WEMAC. The output is a .log file.
4.Creat_feature_maps_wesad.py could generate 2D feature maps using in training process.
5.Then we can start the series of training process. We have general training, LOSO training and RFE training for 3 classes and 2 classes, respectively.

**PLEASE CITE:**
@ARTICLE{10506582,
  author={Sun, Junjiao and Portilla, Jorge and Otero, Andres},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={A Deep Learning Approach for Fear Recognition on the Edge based on Two-dimensional Feature Maps}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Feature extraction;Biomedical monitoring;Emotion recognition;Physiology;Anxiety disorders;Biomarkers;Real-time systems;Affective computing;Fear recognition;Deep learning;Feature selection;Physiological signals;Edge Computing},
  doi={10.1109/JBHI.2024.3392373}}
