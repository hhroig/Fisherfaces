% ------------------------------------------------
% "FLD_Face_Classifier": It calls function "fisherfaces" to get
% the classifiers resulting from a "leave one out" learning splitting. It
% classifies new images from an independet data set. This code calls
% function "Load_Test_Faces" to acquire this independet data set. The face
% recognition procedure here presented calculates distances of every new
% image to the face-space in order to determine wether it is a face or not.
% It also computes the distances to each face-class in order to check if
% new individuals are known or unknonw.
% 
% Harold Antionio Hernández Roig 
% hahernan@est-econ.uc3m.es
% ------------------------------------------------

close all
clear
%% Initialize "fisherfaces" function
show_ave = 0;  % show the average face...
show_4eig = 0; % show the first 4 eigenfaces...
show_roc = 1;  % show the ROC curve for each leave one out

[threshold, TPR, FPR, All_Classes, All_Fisherfaces, class_name, Psi] = ...
    fisherfaces(show_ave, show_4eig, show_roc);

% After Analysis we choose the following set for clasification:

op_set = 1; % fixes which classifier (from all 6 leave-a-face-out) to use
W_opt = All_Fisherfaces{op_set};
Y_classes = All_Classes{op_set};

sigma = threshold(1,op_set); % maximum distance from face-sapace, in row we put the number for corresponding percentile (1 for 100%)

%% Load Test Data Set of New Faces

[subjectID, newFs, file_name] = Load_Test_Faces;
[p,s] = size(newFs); % p: pixels, s: # of subjects in Test

% We do not mean-adjust in FLD!
Test = newFs; 

%% Processing New Faces... dealing with the test set "Test"

% Extract fisherface components of test faces:
Y_test = W_opt'*Test; 

% Distances to Face-Space
dist_fspace = zeros(1,s);
for face = 1:s
    dist_fspace(face) = norm( Test(:,face) - W_opt*Y_test(:,face) ); % difference of input image from its projection onto face-space    
end

% Out the non-faces: those away from face space!
out_no_face = dist_fspace > sigma;
no_faces = file_name(out_no_face);  % then we can have the complete file name of those "no-faces"
file_name(out_no_face) = [];
real_subjectsIDs = subjectID(~out_no_face);
s_real = length(real_subjectsIDs);
no_faces_rate = 1- s_real/s;
Y_test(:,out_no_face) = [];

% Builiding the Matrix D of distances from each individual in 
% Test to each of the 100 classes:

D = zeros(s_real,100); % each d_ij is distance of subject i to class j
for i = 1:s_real
    for j = 1:100
        D(i,j) = norm(Y_test(:,i)-Y_classes(:,j));
    end
end

% Find the face class that minimizes de Eucl. Dist:
[eps_k, location] = min(D,[],2);

estimated_subject = class_name(location);

count_TP = 0;
for m = 1:length(location)
    if strcmp(real_subjectsIDs{m}, estimated_subject{m})
        count_TP = count_TP + 1;
    end
end

well_classified_rate = count_TP/s_real;
wrong_classified_rate = 1- count_TP/s_real; % this is just the FP or wrongly classified subjects!
    
%% Summary of Results:
disp('------------------------------------------------------')
disp('             ----- SUMMARY -----')
disp(['* There were ', num2str(s),' new images to test the algorithm;'])
disp(['and ', num2str(no_faces_rate*100),'% (',num2str(length(no_faces)),'  images) were not detected as faces.'])
disp(['* This reduces the classification to ', num2str(s_real),' faces.'])
disp(['* The rate of wrong classified faces is ', num2str(wrong_classified_rate*100),'% (', num2str(s_real-count_TP),' images) and;'])
disp(['* The rate of well classified subjects is ', num2str(well_classified_rate*100),'% (', num2str(count_TP),' images).' ])
disp(['Evaluation: I wrongly detected as non-faces: ', num2str(length(no_faces) - 5), ' images.', ' From those detected as faces I wrongly classified: ', num2str(s_real-count_TP),' faces.'])
disp(['            Then the true percentage of well classified is: ', num2str(100 -(length(no_faces) - 5 + s_real-count_TP)/205*100), '% .'])
disp('------------------------------------------------------')
