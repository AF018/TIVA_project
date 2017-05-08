%% Utilisation du code Github, prise en main pour application aux bases de donn�es

% Fonction pour aller lire les images dans la base de donn�es
db_name = 'AT&T Database';
[X y w h] = read_images([pwd '/att_data']);
%db_name = 'faces94 Database';
%[X y w h] = read_images([pwd '/faces94']);

% X est la matrice obtenue par concat�nation des images transform�es en
% vecteurs
[n d] = size(X);

% R�cup�ration des classes
c = unique(y);

% Analyse en composante principale
[W mu] = fisherfaces(X, y);

% Affichage des 16 premi�res fisherfaces
figure;
hold on;
title(sprintf('Fisherfaces %s', db_name));
for i=1:min(16,length(c)-1)
  subplot(4,4,i);
  fisherface_i = toGrayscale(W(:,i), w, h);
  imshow(fisherface_i);
  %colormap(jet(256));
  colormap(gray);
  title(sprintf('Fisherface #%i', i));
end