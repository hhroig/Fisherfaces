function [threshold, TPR, FPR, All_Classes, All_Fisherfaces, class_name,...
   all_Psi] = fisherfaces(show_ave, show_4eig, show_roc)

% ------------------------------------------------
% "fisherfaces": Calculates, via leave one out, 6 different
% blocks for face recognition. It divides the data into train and test sets
% in order to correctly determine projections on FLD space of the matrix of training faces.
% This splitting also allows the user to study the capacity of the
% Fisherfaces to classify the test set of images as real faces and return 
% the class to which they belong. The accuracy of the classification is
% studied through the distances of each image to the face-space and to
% every face-class. It firstly applies PCA to reduce dimensionality, then
% performs FLD.
% The data must be in a Folder with name "caras"!
% 
% Harold Antionio Hernández Roig 
% hahernan@est-econ.uc3m.es
% ------------------------------------------------

%% Loading Data
 root = cd;
 cd 'caras'

names = dir('*.bmp');
data=[];
for k=1:length(names)
        disp(['Reading file ' names(k).name]); 
         Im = imread(names(k).name);
         
        % working without colors
        Im_gray = rgb2gray(Im);        
        Im_gray = mat2gray(Im_gray); % converts the same image into a [0,1] gray-scale!
        data(:,k) = Im_gray(:); % every column is a face
        
%         working with colors (not used: grey-scale reduces computational cost and improves accuracy)
%         Im = double(Im);   
%                 
%         red = Im(:,:,1);
%         green = Im(:,:,2);
%         blue = Im(:,:,3);
%         
%         temp = [red(:); green(:); blue(:)];
%         data(:,k) = temp; % every column is a face
end

%% Construction of 3D matrix of faces...

G = data; % matrix of training faces: [G1 G2, G3, ..., GM], every column Gi is a face.
[N,M] = size(G);  % for me this N pixels of image and M subjects times (6) pics per subjects

A3D = zeros(N,6,M/6); % Similiar as A: every row (p) is a pixel, every column in 2nd dimension (t) is a type of face and every number in 3rd dim. (s) represents a subject
for i=0:99
    A3D(:,:,i+1) = G(:,6*i+1:6*(i+1));
end

[p,t,s] = size(A3D); % p = pixels in rows, t = type of face, s = subject

% names of files:
images_names = {names.name};
images_names = reshape(images_names, t, s); % name of files: rows for each face type, cols. for each subject
class_name = cell(1,s);
for i = 1:s
class_name{i} = images_names{1,i}(1:5);
end

%% Leave One Face Out!!!

All_Classes = cell(1,6);
All_Fisherfaces = cell(1,6);
% All_missclass = zeros(1,6);
% All_wellclass = zeros(1,6);
% All_not_faces = zeros(1,6);
% All_unkn = zeros(1,6);

quant_dist_fspace = 1:-0.01:0;
threshold = zeros(length(quant_dist_fspace),t); % minimum allowable distance from face space

TPR = zeros(size(threshold));
FPR = zeros(size(threshold));
all_Psi = zeros(p,t);

for ind_test = 1:t % "leave one face out"
    
    Test = reshape(A3D(:,ind_test,:),p,s); % test set of faces: is a vector due to leave one out
    
    A_train = A3D;
    A_train(:,ind_test,:) = []; % train set of faces in 3D
    A = reshape(A_train, p, (t-1)*100); % train set of faces in 2D
    
    Psi = mean(A,2); % average face of training set ("mu" for Fisherfaces)
    all_Psi(:,ind_test) = Psi;
     
    if show_ave % optional plot of ave-face
    figure
    ave_train_face = reshape(mat2gray(Psi),165,120);
    imshow(ave_train_face)
    title(['Average Face of Training Set: Leave Out Face No.',num2str(ind_test)])
    end
    
    % NON-Mean Adjusted Train and Test (Fisherface Notation)
     X = A_train;
     X_test = Test;
    
    % Mean_adjusted Train for PCA:
    A = A - repmat(Psi,[1,(t-1)*100]);
    
%% Eigenvalues/vectors of Transpose Matrix

L = A'*A; % MxM matrix from which we extract the eigenvectors...

[v, mu_D] = eig(L);

mu = diag(mu_D); % the corresponding eigenvalues
aux = find(mu > 0.00001);

% let's take out the corresponding to mu = 0, using the "aux" vector:
mu = mu(aux);
v = v(:,aux);

u = A*v; % then each column u_i of u is an eigenvector of C = A*A'

% normalize the eigenvectors:
for i = 1:size(u,2)
    u(:,i) = u(:,i)/sqrt(u(:,i)'*u(:,i)); 
end

% let's sort the eigenvalues (then the corresponding eigenvectors)
[mu,I] = sort(mu,'descend');
u = u(:,I);

%% Keep at most k = N-c Principal Components
%  N is the number of images in the train set (500 for leave-a-face-out)
%  and c is the amount of classes (100). This makes Sb non-singular!

% Check the amount of eigenvectors explaining most of the variance:
trace = sum(mu);
K = 0; 
variance_kept = 0;
while variance_kept < 0.95
    K = K+1;
    variance_kept = sum(mu(1:K))/trace;
end

k = min((t-1)*s - s, K); % It's enough with k = N-c but we compare to the K "most important" eigenfaces
%k = 120; % works quite good for k = 120 across models! But the best
%classifier is attained with k = min(...) for leave face no.1 out!

W_pca = u(:,1:k); % we keep in W_pca = U = [u1, ..., uk] the "N-c" features 

% Lets check out the firsts eigenfaces:
if show_4eig % optional plot of first 4 eigenfaces...
figure;
eigenface1 = reshape(mat2gray(W_pca(:,1)),165,120);
subplot(2,2,1),imshow(eigenface1)
title(['1st Eigenface. Leave Out Face No.',num2str(ind_test)])
eigenface2 = reshape(mat2gray(W_pca(:,2)),165,120);
subplot(2,2,2),imshow(eigenface2)
title(['2nd Eigenfaces Leave Out Face No.',num2str(ind_test)])
eigenface3 = reshape(mat2gray(W_pca(:,3)),165,120);
subplot(2,2,3),imshow(eigenface3)
title(['3rd Eigenface. Leave Out Face No.',num2str(ind_test)])
eigenface4 = reshape(mat2gray(W_pca(:,4)),165,120);
subplot(2,2,4),imshow(eigenface4)
title(['4th Eigenface. Leave Out Face No.',num2str(ind_test)])
end


%% Fisher's Linear Discriminant

mean_X = squeeze(mean(X,2));  % Matrix of classes means, i.e. mean_X(:,i) = mu_i, i=1, ..., s(100)
dif_mu = mean_X - Psi; % dif_mu(:,i) = mu_i - mu
proy_M =  W_pca'*dif_mu; %zeros(k, s); % Proyect each mu_i - mu into "face-space": W_pca' * (mu_i - mu) 

SSb = (t-1)*(proy_M*proy_M'); % Instead of Sb, compute SSb = W_pca' * Sb * W_pca 

X_adj = zeros(size(X)); % Each class of X is class-mean-adjusted: forall kx in X_i we have xk-mu_i
for ind_class = 1:s
    X_adj(:,:,ind_class) = X(:,:,ind_class) - repmat(mean_X(:,ind_class),[1,t-1]);
end

W_adj = zeros(k,t-1,s); % projection of each class-mean-adjusted face in "PCA face space"
for slice = 1:s
    W_adj(:,:,slice) = W_pca'*X_adj(:,:,slice);
end

SSw = zeros(k);  % Instead of Sw, compute SSw = W_pca' * Sw * W_pca
for slice = 1:s
    SSw = SSw + W_adj(:,:,slice)*W_adj(:,:,slice)';
end

[W_fld, ~] = eig(SSw\SSb);  % MATLAB says use "SSw\SSb" instead of "inv(SSw)*SSb"

W_opt = W_pca*W_fld;

All_Fisherfaces{ind_test} =  W_opt;

%% Construct the Face Classes (Omega_i)
% note: we have 5 test images per person due to leave one out, 100 indiv. and we can
% calculate Omega_i, i = 1,..., 100 by averaging the results of the fisherface
% representation over a small number of face images of each individual
% (those in the training set)

% We use W_opt' = (W_pca*W_fld)' to extract components in face-space ...
Y = zeros(k,t-1,s); % components of each image in "face space"
for slice = 1:s
    Y(:,:,slice) = W_opt'*X(:,:,slice);
end

% Each column of Y_classes is the mean-vector or class Omega_i, i = 1,..., 100
Y_classes = squeeze(mean(Y,2));

All_Classes{ind_test} =  Y_classes;

% Distances to Face-Space of each
train_dist_fspace = zeros(t-1,s);
for subj = 1:s
for face = 1:t-1
    train_dist_fspace(face,subj) = norm( X(:,face,subj) - W_opt*Y(:,face,subj) ); % difference of input image from its projection onto face-space    
end
end

% Decision threshold theta_eps
threshold(:,ind_test) = quantile(train_dist_fspace(:), quant_dist_fspace); % we save quatiles of distances from face space, later to be used as thresholds for non/unknown faces


%% Processing New Faces... dealing with those left out in "X_test"

% Extract fisherface components of test faces:
Y_test = W_opt'*X_test; 

% Builiding the Matrix D(100 x 100) of distances from each individual in 
% W_test to each of the 100 classes:

D = zeros(s,s); % each d_ij is distance of subject i to class j
for i = 1:s
    for j = 1:s
        D(i,j) = norm(Y_test(:,i)-Y_classes(:,j));
    end
end

% Distances to Face-Space of Test Faces
dist_fspace = zeros(1,s);
for face = 1:s
    dist_fspace(face) = norm( X_test(:,face) - W_opt*Y_test(:,face) ); % difference of input image from its projection onto face-space    
end

% Decision threshold theta_eps
%threshold(:,ind_test) = quantile(dist_fspace, quant_dist_fspace); % we save quatiles of distances from face space, later to be used as thresholds for non/unknown faces

[eps_k, location] = min(D,[],2);

for q = 1:length(quant_dist_fspace)
    temp_theta = threshold(q, ind_test); % let distances of subjects classified as belonging to face-space
    bien_class = 0;
    unkn_subs = 0;
    not_a_face = 0;
    for cuenta = 1:length(location)
        if (cuenta == location(cuenta)) && (eps_k(cuenta) <= temp_theta) && (dist_fspace(cuenta) <= temp_theta)
        bien_class = bien_class + 1;
        elseif eps_k(cuenta) > temp_theta && (dist_fspace(cuenta) <= temp_theta)
            unkn_subs = unkn_subs + 1;
        elseif (dist_fspace(cuenta) > temp_theta)
            not_a_face = not_a_face +1;
        end
    end
    mal_class = s - bien_class - unkn_subs - not_a_face;
TPR(q,ind_test) = bien_class/s; % Well classified
FPR(q,ind_test) = (mal_class + unkn_subs + not_a_face)/s ; % Falsely classified: counts unknown/rongly classified and non-faces
end


if show_roc
figure
plot(quant_dist_fspace,quant_dist_fspace, FPR(:,ind_test),TPR(:,ind_test), 'o')
labels = cellstr( num2str(quant_dist_fspace') );  %' # labels correspond to their order
text(FPR(:,ind_test),TPR(:,ind_test), labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','left')
axis([0 1 0 1])
legend('x = y', 'Rates for \epsilon_{\alpha %} thresholds')
title(['quasi-ROC Space for Different Values of \theta_{\epsilon}: Leave Out Face No.',num2str(ind_test)],'Interpreter','tex')
xlabel('Wrong Classification Rate: unknown/wrongly classified and non-faces')
ylabel('Correct Classification Rate')
end

end % of leave one out
cd(root)
end % of function !!!