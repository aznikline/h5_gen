clear;
close all;
folder = '../DIV2K_HR_x4';
save_root = '../DIV2K_train_part';
savepath = fullfile(save_root ,sprintf('DIV2K_x4_Part%d.h5', 8))

if ~isdir(save_root)
    mkdir(save_root)
end

%% scale factors
scale = 4;

size_label = 96;
size_input = size_label/scale;
stride = 48;


data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

for i = 701 : 800
                image = imread(fullfile(folder,filepaths(i).name));
                if size(image,3)==3
                    %image = rgb2ycbcr(image);
                    image = im2double(image);
                    im_label = modcrop(image, scale);
                    [hei,wid, c] = size(im_label);

                    filepaths(i).name
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            [dx,dy] = gradient(subim_label);
                            gradSum = sqrt(dx.^2 + dy.^2);
                            gradValue = mean(gradSum(:));
                            if gradValue < 0.025
                                continue;
                            end
                            subim_input = imresize(subim_label,1/scale,'bicubic');
                            % figure;
                            % imshow(subim_input);
                            % figure;
                            % imshow(subim_label);
                            count=count+1;
                            data(:, :, :, count) = subim_input;
                            label(:, :, :, count) = subim_label;
                        end
                    end
                end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);