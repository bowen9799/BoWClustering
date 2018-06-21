# Surf-BoW-ColorHist Image Similarity Detection
Input: 1) large number of themed images with portion of false samples (favorably <= 1/2>), 2) ~5 sample classic images; 
Output: 1) images labeled true; 2) images labeled false.

The bigger picture: given a theme under a topic -> crawler fetch images S -> NN to delete off-topic images -> this tool returns true and false samples from S -> resNet / mobileNet trains and classify all of S -> Inception V3 transform learning trains on ~10,000 such S and generate the model -> deploy on cloud server / embed into iOS or Android App.

pipeline: normalization -> SURF extraction -> K-Means clustering & compressing -> BoW cache -> foreach classic image's SURF: dot product in BoW score & RGB color histogram score -> ranking / voting system -> output true & false images.