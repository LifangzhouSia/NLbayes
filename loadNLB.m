function [Stage1st, Stage2nd] = loadNLB(Insy, rnd_str, k1, k2, gamma, beta1, beta2, n1, n2, N1, N2, tau0)

% The pixel number of a Match Window and Search Window
K1 = (2*k1 + 1)^2;
K2 = (2*k2 + 1)^2;
F1 = (2*n1 + 1)^2;
F2 = (2*n2 + 1)^2;

% The Noise Level
h = rnd_str;
h2 = h^2;

[hei, wid] = size(Insy);

% Prepare the padded data
Ipadded = padarray(Insy, [n1+k1, n1+k1], 'symmetric');
image_pad = zeros(size(Ipadded));
count_pad = zeros(size(Ipadded));
use_mask = padarray(zeros(hei, wid), [n1, n1], 'symmetric');

for i=1:hei
    clc;
    fprintf('\tStage1st Estimate, Row %d / %d\n', i, hei);
    for j=1:wid
        i1 = i+ n1+ k1;
        j1 = j+ n1+ k1;
        
        if use_mask(i+n1, j+n1) > 0
            continue
        end
        
        W1 = reshape(Ipadded(i1-k1:i1+k1, j1-k1:j1+k1), K1, 1); %
        
        % The four corner of Search Window
        rmin = i1 - n1;
        rmax = i1 + n1;
        smin = j1 - n1;
        smax = j1 + n1;
        
        % Group the N1 similar patch
        sample_mat = zeros(K1, F1);
        k = 1;
        for r = rmin:1:rmax
            for s = smin:1:smax
                W2 = reshape(Ipadded(r-k1:r+k1 , s-k1:s+k1), K1, 1);
                sample_mat(:, k) = W2;
                k = k + 1;
            end
        end

        distance_mat = mean((sample_mat - repmat(W1, 1, F1)).^2);
        distance_label = sort(distance_mat);
        use_or_not = distance_mat<=distance_label(N1);
        
        sample_use = sample_mat;
        sample_use(:, use_or_not==0) = [];
        
        % the Flat Area 'Trick'
        GroupVar = var(sample_mat(:));
        if GroupVar < (gamma*h2)
            pred_patch = mean(sample_mat, 2);
            for r = rmin:rmax
                for s = smin:smax
                    image_pad(r-k1:r+k1, s-k1:s+k1) = image_pad(r-k1:r+k1, s-k1:s+k1) + reshape(pred_patch, 2*k1+1, 2*k1+1);
                    count_pad(r-k1:r+k1, s-k1:s+k1) = count_pad(r-k1:r+k1, s-k1:s+k1) + 1;
                    use_mask(r-k1, s-k1) = use_mask(r-k1, s-k1) + 1;
                end
            end
            continue
        end
        % Flat Area 'Trick' is end
        
        average_mat = mean(sample_use, 2);
        sample_dec = sample_use - repmat(average_mat, 1, size(sample_use,2));
        conv_mat = (1/(size(sample_dec,2)-1)) * sample_dec * sample_dec';
        
        sample_denoised = sample_use - h2*ConvarianceInverse(conv_mat)*sample_dec;
        % NL_Bayes denoise module
        new_sample_mat = sample_mat;
        new_sample_mat(:, use_or_not) = sample_denoised;
        k = 1;
        for r = rmin:rmax
            for s = smin:smax
                if use_or_not(k) == 0
                    k = k + 1;
                    continue
                end
                image_pad(r-k1:r+k1, s-k1:s+k1) = image_pad(r-k1:r+k1, s-k1:s+k1) + reshape(new_sample_mat(:,k), 2*k1+1, 2*k1+1);
                count_pad(r-k1:r+k1, s-k1:s+k1) = count_pad(r-k1:r+k1, s-k1:s+k1) + 1;
                use_mask(r-k1, s-k1) = use_mask(r-k1, s-k1) + 1;
                k = k + 1;
            end
        end
    end
end

Stage1st = image_pad(1+n1+k1:end-n1-k1, 1+n1+k1:end-n1-k1) ./ count_pad(1+n1+k1:end-n1-k1, 1+n1+k1:end-n1-k1);

%% Stage2st --- Bayesian Final Estimate

Ipadded = padarray(Insy, [n2+k2, n2+k2], 'symmetric');
S1padded = padarray(Stage1st, [n2+k2, n2+k2], 'symmetric');

image_pad = zeros(size(Ipadded));
count_pad = zeros(size(Ipadded));

use_mask = padarray(zeros(hei, wid), [n2, n2], 'symmetric');

for i=1:hei
    clc;
    fprintf('\tStage2nd Estimate, Row %d / %d\n', i, hei);
    for j=1:wid
        i1 = i+ n2+ k2;
        j1 = j+ n2+ k2;
        
        if use_mask(i+n2, j+n2) > 0
            continue
        end
        
        W1 = reshape(S1padded(i1-k2:i1+k2, j1-k2:j1+k2), K2, 1); 
        W1_noisy = reshape(Ipadded(i1-k2:i1+k2, j1-k2:j1+k2), K2, 1); 
        
        rmin = i1 - n2;
        rmax = i1 + n2;
        smin = j1 - n2;
        smax = j1 + n2; 
        
        sample_mat = zeros(K2, F2);
        sample_mat_noisy = zeros(K2, F2);
        k = 1;
        for r = rmin:1:rmax
            for s = smin:1:smax
                W2 = reshape(S1padded(r-k2:r+k2, s-k2:s+k2), K2, 1);
                W2n = reshape(Ipadded(r-k2:r+k2, s-k2:s+k2), K2, 1);
                sample_mat(:, k) = W2;
                sample_mat_noisy(:, k) = W2n;
                k = k+1;
            end
        end
        
        % the Flat Area 'Trick'
        GroupVar = var(sample_mat_noisy(:));
        if GroupVar < (gamma*h2)
            pred_patch = mean(sample_mat, 2);
            for r = rmin:rmax
                for s = smin:smax
                    image_pad(r-k2:r+k2, s-k2:s+k2) = image_pad(r-k2:r+k2, s-k2:s+k2) + reshape(pred_patch, 2*k2+1, 2*k2+1);
                    count_pad(r-k2:r+k2, s-k2:s+k2) = count_pad(r-k2:r+k2, s-k2:s+k2) + 1;
                    use_mask(r-k2, s-k2) = use_mask(r-k2, s-k2) + 1;
                end
            end
            continue
        end
        % the Flat Area 'Trick' is end
        
        distance_mat = mean((sample_mat - repmat(W1, 1, F2)).^2);
        distance_label = sort(distance_mat);
        use_or_not = (distance_mat<= min(distance_label(N2), tau0));
        % the N2 similar or the patch distace below tau0 is used
        
        sample_use = sample_mat;
        sample_use(:, use_or_not==0) = [];
        
        average_mat = mean(sample_use, 2);
        sample_dec = sample_use - repmat(average_mat, 1, size(sample_use,2));
        conv_mat = (1/(size(sample_dec,2)-1)) * sample_dec * sample_dec';
        
        sample_dec2 = sample_mat_noisy(:, use_or_not) - repmat(average_mat, 1, sum(use_or_not));
        sample_denoised = repmat(average_mat, 1, size(sample_dec2, 2)) + conv_mat*ConvarianceInverse(conv_mat+h2*eye(K2))*sample_dec2;
        % NL_Bayes denoise module
        new_sample_mat = sample_mat;
        new_sample_mat(:, use_or_not) = sample_denoised;
        k = 1;
        for r = rmin:rmax
            for s = smin:smax
                if use_or_not(k) == 0
                    k = k + 1;
                    continue
                end
                image_pad(r-k2:r+k2, s-k2:s+k2) = image_pad(r-k2:r+k2, s-k2:s+k2) + reshape(new_sample_mat(:,k), 2*k2+1, 2*k2+1);
                count_pad(r-k2:r+k2, s-k2:s+k2) = count_pad(r-k2:r+k2, s-k2:s+k2) + 1;
                use_mask(r-k2, s-k2) = use_mask(r-k2, s-k2) + 1;
                k = k + 1;
            end
        end
%         disp('ye')
    end
end
Stage2nd = image_pad(1+n2+k2:end-n2-k2, 1+n2+k2:end-n2-k2) ./ count_pad(1+n2+k2:end-n2-k2, 1+n2+k2:end-n2-k2);
% Stage2nd = Stage1st;
end

function invCp = ConvarianceInverse(Cp)
% under the circumstance the CCD is saturated, the local variance can be
% below the Noise Level, which causes the convariance matrix to be sigular
% if this happened, we output a empty matrix
% under the ordinary cases, a cholesky decomposition method is used
[d, dtest] = size(Cp);
if d ~= dtest
    error('Error: Square Matrix is Needed');
end
rankCp = rank(Cp);
if rankCp < size(Cp, 1)
    invCp = zeros(size(Cp, 1));
else
    R = chol(Cp);
    Rex = [R, eye(d)];
    for k = d:-1:1
        Rex(k, :) = Rex(k, :) ./ Rex(k, k);
        for g = k-1:-1:1
            Rex(g, :) = Rex(g, :) - Rex(k, :).*(Rex(g, k));
        end
    end
    invR = Rex(:, d+1:end);
    invCp = invR * invR';
end
end
    

