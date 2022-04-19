# Helmet Violation Results
## Helmet Violation Detection And Identification Performance
![Helmet_Violations](https://user-images.githubusercontent.com/56833595/150500420-6199a4b2-df0c-4e49-8e9e-065eaf4e8e28.png)

## Base Model
Weights for all the base models trained on our training set can be found [here].

## ROI Extraction Approach
Go to ROI Extraction folder and choose the scipt cooresponding to the association approach

## Steps to be followed for result replication
Step 1: Download the corresponding Rider-Motorcycle weights(RM_weights), Helmet-No-Helmet weights(HNH_weights) and test dataset <br>
Step 2: Get the results.json file on the RM_weights and image data using [![Train Custom Model In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing)<br>
Step 3: Convert results.json to different (.txt) files using  [![Json_to_txt.py](https://github.com/CVIT-Mobility/tripple-rider-violations/blob/main/missc/json_to_txt.py)]<br>
Step 4: use corresponding [![ROI Extraction](https://github.com/CVIT-Mobility/tripple-rider-violations/tree/main/results/Helmet%20Violations/ROI%20Extraction)] approach to get the image crops<br>
Step 5: Get the results_1.json file on the HNH_weights and crop-image data using  [![Train Custom Model In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing)<br>
Step 6: Convert results_1.json to images using  [![Json_to_img.py](https://github.com/CVIT-Mobility/tripple-rider-violations/blob/main/missc/json_to_img.py)]<br>
