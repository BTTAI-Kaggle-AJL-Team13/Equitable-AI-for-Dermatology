# Equitable AI for Dermatology

## Project Overview
The Kaggle competition, in partnership with the Break Through Tech AI Program and the Algorithmic Justice League, addresses bias in dermatology AI tools, which often underperform for individuals with darker skin tones due to a lack of diverse training data. Participants are tasked with developing inclusive machine learning models to ensure equitable and accurate dermatological diagnoses.

This challenge has a significant real-world impact. Misdiagnoses and delayed treatments disproportionately affect underserved communities, worsening health disparities. By creating fairer AI models, participants can contribute to improving healthcare outcomes and promoting algorithmic justice in medical technology.

## Dataset
The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set. We used a subset in order to create a more manageable and hopefully satisfying classification problem, while trying to maintain some of the representation issues surfaced by the full set.

## Data Preprocessing
Our team performed various steps to ensure the data was clean, uniformly sized, and ready for model input.
- Imported essential libraries, including Keras, PIL, NumPy, and scikit-learn
- Extracted image dimensions and RGB pixel values from JPEG files in the training directory
- Resized images to 128x128 pixels and normalized pixel values to the range [0,1] to standardize input for the model
- Filtered out samples labeled as "Wrongly Labelled"
- One-hot encoded categorical variables

## Visualizations
<img src="https://github.com/user-attachments/assets/2272bd20-0025-40d7-acd2-68f2375c79d6" alt="Fitzpatrick Scale" width="300" height="200">


## Team Members
- Ashley Bao (ashleybao)
- Aimee Hong (aimeehong1)
- Varsha Athreya (varsha487)
- Anissa Patel (anissakp)
- Sally Lee (delee5695)
- Sneha Sriram (snehasriram1013)
