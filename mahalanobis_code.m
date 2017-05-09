%% Code pour la reconnaissance de visages avec la distance de Mahalanobis

% Inutlisable en pratique pour faces94 : trop de donnees

% Fonction pour aller lire les images dans la base de données
[X y w h] = read_images([pwd '/att_data']);
%[X y w h] = read_images([pwd '/faces94']);

% X est la matrice obtenue par concaténation des images transformées en
% vecteurs
[n d] = size(X);

% Calcul de la matrice de covariance intra personnelle
% On utilise la formulation alternative de Xun Xu et Thomas Huang dans
% "Face Recognition with MRC Boosting"
intra_W = zeros(n, n);
for i = 1:max(y)
    y_test = double(y == i);
    intra_W = intra_W + y_test' * y_test;
end
extra_W = 1 - intra_W;

e = ones(n, 1);

intra_sig = 2 * X' * (diag(intra_W * e) - intra_W) * X;
extra_sig = 2 * X' * (diag(extra_W * e) - extra_W) * X;

% Appel de l'image a classer
%Q = X(1,:); % premiere image de la base de donnees
%Q_rgb = double(imread('test_2.jpg')); % Pour faces94
Q_rgb = double(imread('test_1.jpg')); % Pour At&T
Q = 0.2126*Q_rgb(:,:,1)+0.7152*Q_rgb(:,:,2)+0.0722*Q_rgb(:,:,3);
Q = reshape(Q,1,w*h);
dist_i = repmat(Q, n, 1) - X;

% Calcul des log-vraisemblances intrapersonnelles
eigen_values = eig(intra_sig);
intra_vol_log = (d/2) * log(2 * pi) + 0.5 * sum(log(abs(eigen_values)));
mahala_dist_log = dist_i * inv(intra_sig) * dist_i'; % ca risque d'etre faux ici avec la manipulation de matrice au lieu de vecteurs
mahala_dist_log = diag(mahala_dist_log);
intra_log_lklhd = - intra_vol_log - mahala_dist_log;