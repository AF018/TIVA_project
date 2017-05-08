%% Utilisation du code Github, prise en main pour application aux bases de données

% faces94 tres long a charger, contrairement a att_data
% Les resultats sont plus significatifs avec AT&T, on reconnait certaines
% caracteristiques des visages contrairement a faces94.
% Ceci peut s'expliquer par le fait que la LDA permet de souligner ce qui
% distingue les classes. Or, avec beaucoup plus de personnes, il est
% difficile de comprendre a l'oeil nu le contenu des fisherfaces car les
% differences sont plus subtiles

% Fonction pour aller lire les images dans la base de données
%db_name = 'AT&T Database';
%[X y w h] = read_images([pwd '/att_data']);
db_name = 'faces94 Database';
[X y w h] = read_images([pwd '/faces94']);

% X est la matrice obtenue par concaténation des images transformées en
% vecteurs
[n d] = size(X);

% Récupération des classes
c = unique(y);

% Analyse en composante principale
[W mu] = fisherfaces(X, y);

% Affichage des 16 premières fisherfaces
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

% Appel de l'image a reconstruire
%Q = X(1,:); % premiere image de la base de donnees
Q_rgb = double(imread('test_2.jpg')); % Pour faces94
%Q_rgb = double(imread('test_1.jpg')); % Pour At&T
Q = 0.2126*Q_rgb(:,:,1)+0.7152*Q_rgb(:,:,2)+0.0722*Q_rgb(:,:,3);
Q = reshape(Q,1,w*h);

% Projection d'un visage sur chacune des fisherfaces
steps = 1:min(16, length(c)-1) ;
figure;
hold on ;
title(sprintf('Fisherfaces Reconstruction %s',db_name)) ;
for i = 1:min(16,length(steps))
    subplot(4,4,i);
    numEv = steps(i) ;
    P = project(W(:,numEv),X(1,:),mu);
    R = reconstruct(W(:,numEv),P,mu);
    comp = toGrayscale(R,w,h);
    imshow(comp);
    title(sprintf('Fisherface #%i',numEv));
end
