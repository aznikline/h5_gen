clear;
close all;
folder_img = '../JPEGImages';
folder_label = '../SegmentationClassAug';
save_root = '../VOC_train_label160_Scales_HDF5';
hr_root = '../Test/HR_train_patches';
lr_root = '../Test/LR_train_patches';
label_root = '../Test/Label_train_patches';
list = '../ImageSets/SSSRNet/train_aug.txt';

name_lists=importdata(list);

if ~isdir(save_root)
    mkdir(save_root)
end
if ~isdir(hr_root)
    mkdir(hr_root)
end
if ~isdir(lr_root)
    mkdir(lr_root)
end
if ~isdir(label_root)
    mkdir(label_root)
end
%% scale factors
scale = 4;

size_label = 160;
size_input = size_label/scale;
stride = 80;
downsizes = [1, 0.7, 0.5];




%% generate data
filepaths = [];


length(name_lists)
parts=8;
batch = ceil(length(name_lists)/parts)

for n=1:parts
lr = zeros(size_input, size_input, 3, 1);
hr = zeros(size_label, size_label, 3, 1);
label = zeros(size_input, size_input, 1, 1);
count = 0;
margain = 0;
for i = (n-1)*batch +1 : min(length(name_lists), n*batch)
            for downsize = 1:length(downsizes)     
                image = imread(fullfile(folder_img,[name_lists{i} '.jpg']));
                image_label = imread(fullfile(folder_label,[name_lists{i} '.png']));
                if size(image,3)==3
                    %image = rgb2ycbcr(image);
                    image = im2double(image);
                    image_label = imresize(image_label,downsizes(downsize),'nearest');
                    image = imresize(image,downsizes(downsize),'bicubic');
                    im_hr = modcrop(image, scale);
                    im_label = modcrop(image_label, scale);
                    [hei,wid, c] = size(im_hr);

                    [name_lists{i} '.jpg']
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_hr = im_hr(x : x+size_label-1, y : y+size_label-1, :);
                            [dx,dy] = gradient(subim_hr);
                            gradSum = sqrt(dx.^2 + dy.^2);
                            gradValue = mean(gradSum(:));
                            if gradValue < 0.025
                                continue;
                            end
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            subim_label = imresize(subim_label,1/scale,'nearest');
                            subim_input = imresize(subim_hr,1/scale,'bicubic');
                            %imwrite(subim_hr, fullfile(hr_root,[name_lists{i} sprintf('_%d-%d_hr.png', x, y)]));
                            %imwrite(subim_label, fullfile(hr_root,[name_lists{i} sprintf('_%d-%d_label.png', x, y)]));
                            %imwrite(subim_input, fullfile(hr_root,[name_lists{i} sprintf('_%d-%d_lr.png', x, y)]));
                            %figure;
                            %imshow(subim_input);
                            %figure;
                            %imshow(subim_label);
                            count=count+1;
                            lr(:, :, :, count) = subim_input;
                            hr(:, :, :, count) = subim_hr;
                            label(:, :, :, count) = subim_label;
                        end
                    end
                    end
                end
end

order = randperm(count);
lr = lr(:, :, :, order);
hr = hr(:, :, :, order); 
label = label(:, :, :, order);

%% writing to HDF5
savepath = fullfile(save_root ,sprintf('VOC_x4_Part%d.h5', n))
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batch_lrs = lr(:,:,:,last_read+1:last_read+chunksz); 
    batch_hrs = hr(:,:,:,last_read+1:last_read+chunksz);
    batch_labels = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('lr',[1,1,1,totalct+1],'hr',[1,1,1,totalct+1], 'label', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_3outputs(savepath, batch_lrs, batch_hrs, batch_labels, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
end