% demo for a matlab implementation of Non-local Bayes Algorithm
% Lebrun M, Buades A, Morel J, et al. A Nonlocal Bayesian Image Denoising Algorithm[J]. 
% Siam Journal on Imaging Sciences, 2013, 6(3): 1665-1688.

%% data prepare

I = double(imread('cameraman.tif'));
noise_level = 5;
Insy = I + noise_level*randn(size(I));

%% demo

[Stage1st, Stage2nd] = useNLB(Insy, -1);
imshow([Stage1st Stage2nd], [])

PSNR1st = 10*log10(255^2/mean((Stage1st(:) - I(:)).^2));
PSNR2nd = 10*log10(255^2/mean((Stage2nd(:) - I(:)).^2));
fprintf('\n\tStage1st PSNR is %.2f db\n\tStage2nd PSNR is %.2f db\n', PSNR1st, PSNR2nd)