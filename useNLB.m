function [Stage1st, Stage2nd] = useNLB(Insy, rnd_str)

% if rnd_str is given or estimated through a PCA based method
% the Noise Level Estimation program is modified to fit gray image
if rnd_str == -1
    [rnd_str] = NoiseLevelGrayImage(Insy);
    fprintf('\tAdaptive noise estimation enabled, the Noise Level is %.1f\n', rnd_str);
else
    fprintf('\tthe Noise Level is %.1f\n', rnd_str);
end

% the final chosen values for all parameters,depending on the level of the
% noise, in the article¡¶Implementation of the ¡°Non-Local Bayes¡± (NL-Bayes)
% Image Denoising Algorithm¡·
if rnd_str < 20
    k1 = 3;
    k2 = 3;
    gamma = 1.05;
    beta1 = 1.0;
    beta2 = 1.2;
    n1 = 10;
    n2 = 10;
    N1 = 150;
    N2 = 150;
    tau0 = 1e10;
elseif rnd_str >= 20 & rnd_str < 50
    k1 = 5;
    k2 = 3;
    gamma = 1.05;
    beta1 = 1.0;
    beta2 = 1.2;
    n1 = 35;
    n2 = 21;
    N1 = 60;
    N2 = 30;
    tau0 = 4;
elseif rnd_str >= 50 & rnd_str < 70
    k1 = 7;
    k2 = 5;
    gamma = 1.05;
    beta1 = 1.0;
    beta2 = 1.0;
    n1 = 49;
    n2 = 35;
    N1 = 90;
    N2 = 60;
    tau0 = 4;
elseif rnd_str >= 70
    k1 = 7;
    k2 = 7;
    gamma = 1.05;
    beta1 = 1.0;
    beta2 = 1.0;
    n1 = 49;
    n2 = 49;
    N1 = 90;
    N2 = 90;
    tau0 = 4;
end

timeFirst = now;
[Stage1st, Stage2nd] = loadNLB(Insy, rnd_str, k1, k2, gamma, beta1, beta2, n1, n2, N1, N2, tau0);
estimate_elapsed_time = 86400*(now-timeFirst);

% Showing running time
fprintf('Filtering completed: %.1f sec', estimate_elapsed_time);

end