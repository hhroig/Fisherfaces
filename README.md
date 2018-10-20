# Assignment 2 : Face Recognition Using   Fisherfaces.

### Introduction

This work deals with face recognition using Fisher's Linear Discriminant Analysis (FLD). This approach consists on projecting the face images into a low-dimensional subspace. This is achieved by selecting the projection matrix $(W_{opt})$ in such a way that the ratio of the between-class scatter and the within-class scatter is maximized. As images are mostly in a high dimensional space (too many pixels), Principal Components Analysis (PCA) is performed as a first step to reduce the dimension. This methodology we call the "fisherface approach" [1] shows better performance than using just "eigenfaces" [2].

### Codes

- *FLD_Face_Classifier*: this would be the main script from which you can call the next two functions and obtain the classifier and classification of new images. There are already some fixed parameters to achieve the best face classifier.
- *fisherfaces*: this function has all Steps from the proposed algorithm and is the central core of this report. The dataset (all 600 original images) must be in a subfolder named "caras".
- *Load_Test_Faces*: loads new data, the ones from folder "retiras" in Aula Global. The new independent dataset should be pasted in a subfolder named "test".

### References

[1] P.N. Belhumeur, J.P. Hespanha, and D.J. Kriegman. Eigenfaces vs. Fisherfaces: recognition using class specific linear projection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(7):711–720, jul 1997.

[2] Matthew Turk and Alex Pentland. Eigenfaces for Recognition. Journal of Cognitive Neuroscience, 3(1):71–86, jan 1991.
