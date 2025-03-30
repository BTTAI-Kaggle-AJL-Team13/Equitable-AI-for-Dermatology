# Equitable AI for Dermatology

## **🎯 Project Highlights**

* Developed a convolutional neural network (CNN) model and utilized MobileNet for transfer learning to improve the classification of skin conditions.
* Used data augmentation and selective unfreezing on MobileNet layers to improve accuracy
* Achieved an F1-score of approximately 0.59 and ranked 19th out of 74 participating teams on the final Kaggle Leaderboard.

---

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

## **🏗️ Project Overview**

The Kaggle competition, in partnership with the Break Through Tech AI Program and the Algorithmic Justice League, addresses bias in dermatology AI tools that often underperform for individuals with darker skin tones due to a lack of diverse and comprehensive training data. Participants are tasked with developing inclusive machine learning models to ensure equitable and accurate dermatological diagnosis using limited medical data and image classification.

This challenge has a significant real-world impact since underperforming AI dermatology tools can lead to diagnostic errors and delayed treatments that disproportionately affect underserved communities and worsen health disparities. By creating more equitable AI models, participants can contribute to improving healthcare outcomes and promoting algorithmic justice in medical technology.

---

## **👩🏽‍💻 Setup & Execution**

* To clone the repository, click the green "<> Code" button in the top right corner. 
* You will need to install the following packages for the code to run. You can use the command ```pip install <package-name>```.
```bash
# Data manipulation and processing
pip install pandas
pip install numpy

# Visualization
pip install matplotlib
pip install seaborn

# Keras and TensorFlow for deep learning
pip install keras
pip install tensorflow

# Image handling
pip install pillow

# Sklearn for metrics and preprocessing
pip install scikit-learn

# Utilities
pip install tqdm  # For progress bars

# MobileNet and other Keras/TensorFlow components
pip install tensorflow  # To include MobileNet and other Keras features
```
 1. **pandas**: For data manipulation.
 2. **numpy**: For numerical operations.
 3. **os**: For interacting with the operating system (part of Python's standard library).
 4. **matplotlib**: For creating plots and visualizations.
 5. **seaborn**: For statistical data visualization.
 6. **keras**: For building neural network models (used with TensorFlow).
 7. **tensorflow**: For deep learning, which includes Keras as part of its API.
 8. **Pillow**: For image processing.
 9. **scikit-learn**: For machine learning, metrics, and preprocessing.
 10. **tqdm**: For progress bars in loops.
 11. **glob**: For finding pathnames matching a specified pattern (part of Python's standard library).
 12. **MobileNet**: Part of TensorFlow's Keras applications.
 13. **OneHotEncoder**: From `sklearn.preprocessing`, used for one-hot encoding.
 14. **LabelEncoder**: From `sklearn.preprocessing`, used for encoding labels.
* To run the scripts, use Google Colab or Jupyter Lab, or any other IDE for editing python notebooks.
* To access the dataset, use the Kaggle competition website, linked at the top of this file.
* Make sure to change any file names and directory paths in the python notebook to the respective development environment you opt to use.
---

## **📊 Data Exploration**

### Dataset
The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set. This competition subset was utilized in order to create a more manageable and satisfying classification problem, while trying to maintain some of the representation issues surfaced by the full set.

The dataset included the following variables: 
md5hash (string) is the unique identifier
* fitpatric_scale (int) is a nominal int in range [-1, 0) and [1, 6] indicating self-described Fitzpatrick Skin Type (FST). A larger FST indicates a darker skin tone.
* fitzpatrick_centaur (int) nominal int in the range [-1, 0) and [1, 6] indicating FST assigned by Centaur Labs, a medical data annotation firm. A larger FST value indicates a darker skin tone
* label (string) is our dependent variable, indicating the medical diagnosis with 21 possible conditions
* nine_partition_label (string) is a categorical variable indicating one of nine diagnostic categories
  * Inflammatory
  * Malignant-epidermal
  * Malignant-melanoma
  * Benign-epidermal
  * Benign-dermal
  * Malignant-cutaneous-lymphoma
  * Malignant-dermal
* three_partition_label (string) is a categorical variable indicating one of three diagnostic categories
  * Malignant
  * Non-neoplastic
  * Benign
* qc (int) a nominal int for quality control check by a Board-certified dermatologist.
  * 1: Diagnostic - the image shows a good example of the skin condition
  * 2: Characteristic - the image shows something that could be the skin condition, but isn't diagnostic
  * 3: Wrongly labelled - the image shows something that is definitely not the labeled condition
  * 4: Other
  * 5: Potentially - not clearly diagnostic, but not necessarily mislabeled, further testing would be required
* ddi_scale (int) A column used to reconcile this dataset with another dataset (may not be relevant)

Missing values: this dataset included only one variable with missing values, qc, which is an important indicator for understanding whether the image is adequately diagnostic of the skin condition. Of our 2860 observations, we had values for qc in only 90 observations, and 30 observations in the test dataset. Of those 90 observations, 4 were wrongly labelled with images that clearly did not pertain to the labelled skin condition. These observations were removed from our model training, but it is unclear how many of the 2770 observations with missing qc values were also wrongly labelled. Furthermore, the test dataset excluded the nine_partition_label and three_partition_label, which are often unavailable until some sort of medical examination.

### Data Preprocessing
Our team performed various steps to ensure the data was clean, uniformly sized, and ready for model input.
- Imported essential libraries, including Keras, PIL, NumPy, and scikit-learn
- Extracted image dimensions and RGB pixel values from JPEG files in the training directory
- Resized images to a consistent size and normalized pixel values to the range [0,1] to standardize input for the model
- Filtered out samples labeled as "Wrongly Labelled"
- One-hot encoded categorical variables

### Visualizations
Distribution of skin types: The Fitzpatrick Scale graph shows the distribution of skin types present in the dataset using Fitzpatrick Skin Types, which range from 1 for the lightest skin tones to 7 for the darkest skin tones. This allows us to assess the diversity of skin tones represented, which is skewed toward lighter skin tones. People with an FST of 4 or larger are a significant minority in the population, with 698 of the 2860 observations, or just less than 25% of the population. Since darker skin tones are such a small percentage of the dataset, our model will have less training data and will likely perform worse in future cases for people with dark skin tones. 

<img src="https://github.com/user-attachments/assets/2272bd20-0025-40d7-acd2-68f2375c79d6" alt="Fitzpatrick Scale" width="275" height="200">

Distribution of skin conditions: Of the 21 skin conditions represented in our dataset, squamous cell carcinoma and basal cell carcinoma are the most common, comprising about 25.7% of the data. Acne, the most common type of skin condition, is only the 7th most represented skin condition in the data, making up about 4.5% of the observations. For many of the less common skin conditions in the dataset, there are only about 50 observations, or less than 2% of the dataset, for the model to learn from. 

![image](https://github.com/user-attachments/assets/0f71b96f-cc20-4b61-a744-c301f2e47959)

---

## **🧠 Model Development**

* Applied an 80% train and 20% test split.
* Trained a baseline CNN on pictures and classifications in the training set. Achieved an F1-score of ~0.2 in Kaggle.
* Fine-tuned a pre-trained CNN model (Resnet and MobileNet) by implementing transfer learning.
* Augmented data from underrepresented skin conditions and increased the number of outer layers in the model.
* Achieved an accuracy of ~90% during training on the validation data and an improved F1-score of ~0.59 in Kaggle. 

---

## **📈 Results & Key Findings**

* Placed 19/74 on Kaggle Competition Leaderboard
* The following graph shows our accuracy and loss across the epochs when training. The model achieved around 90% accuracy on the validation dataset after 15 epochs of training. Also note that even once 100% accuracy was reached on the training dataset, around epoch 4, the validation accuracy continued to increase until around epoch 10.
![curve_loss](https://github.com/user-attachments/assets/b6016d22-bccd-4713-bb21-1ac0d2790b84)
* Below is the confusion matrix for the predictions made in our validation dataset.
![confusion](https://github.com/user-attachments/assets/47240209-53ae-4865-8450-7c9953866e39)
* The following graph is the precision-recall curve for our dataset. 
![prec_recall](https://github.com/user-attachments/assets/a5e7539f-3a66-47c9-b109-63e569e8ae96)
* The following figure shows the performance of the model on the entire provided training dataset, including the data we trained on and validated on. Our training dataset accuracy reached 100%, so we can assume that the differences in accuracy that this plot displays reflect the differences in validation accuracy. We can see that at either end of the fitzpatrick scale, the accuracy does decrease.
![acc_comparison](https://github.com/user-attachments/assets/26aebdc8-b685-46ce-aaf3-79277d4bf2c5)
* Note: In the above figures, the class number to skin condition conversion used is:
  * Class 0: Prurigo Nodularis
  * Class 1: Basal Cell Carcinoma Morpheiform
  * Class 2: Keloid
  * Class 3: Basal Cell Carcinoma
  * Class 4: Seborrheic Keratosis
  * Class 5: Eczema
  * Class 6: Folliculitis
  * Class 7: Squamous Cell Carcinoma
  * Class 8: Actinic Keratosis
  * Class 9: Mycosis Fungoides
  * Class 10: Acne Vulgaris
  * Class 11: Dyshidrotic Eczema
  * Class 12: Melanoma
  * Class 13: Epidermal Nevus
  * Class 14: Malignant Melanoma
  * Class 15: Pyogenic Granuloma
  * Class 16: Dermatofibroma
  * Class 17: Kaposi Sarcoma
  * Class 18: Acne
  * Class 19: Dermatomyositis
  * Class 20: Superficial Spreading Melanoma SSM
    
---

## **🖼️ Impact Narrative**
1. To address model fairness, we leveraged data augmentation techniques to account for training dataset imbalances. This creates a larger dataset with random permutations of flipped, rotated, cropped, and stretched images. We also used a validation set to assess model performance across different skin tones. Lastly, we employed data visualization techniques to better understand the accuracy of our predictions across different skin tones. This allowed us to keep track of if our model has any inherent biases due to gaps in the data.
2. This work could have an impact in the healthcare industry. AI in general could make access to healthcare and diagnoses more available at lower costs, increasing equity in the healthcare industry. While official diagnoses should be made with a professional, machine learning algorithms such as the one we developed can allow patients a way of understanding their condition without having to go through the process of consulting with a medical professional.

---

## **🚀 Next Steps & Future Improvements**

* While we did address the class imbalances using oversampling, our model still performs worse on the both ends of the Fitzpatrick scale. We could work to improve this by having more representative datasets, so our model would be able to have more data to train on.
* With more time and resources, we would fine tune more layers of the MobileNet model and test other transfer learning models. We chose to implement MobileNet because previous research showed that MobileNet performed well for skin condition recognition compared to other models [1]. We would also train on more data to increase the accuracy of our model.
* In addition to using transfer learning - which is the method that resulted in the highest F1 score - we also implemented a CNN from scratch. While we were able to train the model successfully, it took a longer time and did not generalize well. This could be something to further research though.

---

## **📄 References & Additional Resources**

1. Velasco, J. S., Catipon, J. V., Monilar, E. G., Amon, V. M., Virrey, G. C., & Tolentino, L. K. S. (2023). Classification of skin disease using transfer learning in convolutional neural networks. *International Journal of Emerging Technology and Advanced Engineering, 13*(4), 1–7. https://doi.org/10.46338/ijetae0423_01

---

## **👥 Team Members**
| Ashley Bao | @ashleybao | Co-led team, preprocessed data, built baseline CNN model |

| Aimee Hong | @aimeehong1 | Co-led team, conducted exploratory data analysis, one hot encoded categorical variables, and built a baseline CNN model |

| Varsha Athreya | @varsha487 | Preprocessed data, fine-tuned pre-trained CNN model using transfer learning, calculated evaluation metrics |

| Anissa Patel | @anissakp | Researched pre-processing and model selection, built baseline CNN model |
