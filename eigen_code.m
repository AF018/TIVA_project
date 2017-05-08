%% Utilisation du code Github, prise en main pour application aux bases de donnees

% Fonction pour aller lire les images dans la base de données
%[X y w h] = read_images([pwd '/att_data']);
[X y w h] = read_images([pwd '/faces94']);

% X est la matrice obtenue par concaténation des images transformées en
% vecteurs
[n d] = size(X);

% Analyse en composante principale
[W mu] = pca(X, y, n);

% Affichage après normalisation des vecteurs propres et passage en niveaux
% de gris
figure;
hold on;
title('Eigenfaces (AT&T Facedatabase)');
for i=1:min(16,n)
    subplot(4,4,i);
    eigenface_i = toGrayscale(W(:,i), w, h);
    imshow(eigenface_i);
    %colormap(jet(256));
    colormap(gray);
    title(sprintf('Eigenface #%i', i));
end

% Appel de l'image a reconstruire
%Q = X(1,:); % premiere image de la base de donnees
Q_rgb = double(imread('test_2.jpg')); % Pour faces94
%Q_rgb = double(imread('test_1.jpg')); % Pour At&T
Q = 0.2126*Q_rgb(:,:,1)+0.7152*Q_rgb(:,:,2)+0.0722*Q_rgb(:,:,3);
Q = reshape(Q,1,w*h);

% Reconstruction
steps = 10:20:min(n,320);
figure;
hold on;
title('Reconstruction');
for i=1:min(16, length(steps))
    subplot(4,4,i);
    numEvs = steps(i);
    P = project(W(:,1:numEvs), Q, mu);
    R = reconstruct(W(:,1:numEvs),P,mu);
    comp = toGrayscale(R, w, h);
    imshow(comp);
    title(sprintf('%i Eigenvectors', numEvs));
end