1st: Copy train dataset in subfolder "caras"
2nd: Copy test dataset in subfolder "test"
3rd and Final: Call function FLD_Face_Classifier.

Dependencies:

* FLD_Face_Classifier: this would be the main script from which you can call the next two functions and obtain the classifier and classification of new images. There are already some fixed parameters to achieve the best face classifier.

    * fisherfaces: this function has all Steps from the proposed algoritm and is the central core of this work. The dataset (all 600 original images) must be in a subfolder named ``caras''.
    
    * Load_Test_Faces: loads new data, the ones from folder ``retiradas'' in Aula Global. The new independent dataset should be pasted in a subfolder named ``test''.