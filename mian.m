%% Read .mha file -> build image
% start ....
clc;
close();
im_path = '.\0001\VSD.Brain.XX.O.MR_Flair\VSD.Brain.XX.O.MR_Flair.684.mha';
valid_im_path = '.\0001\VSD.Brain_3more.XX.XX.OT\VSD.Brain_3more.XX.XX.OT.6560.mha';

% current non validated image
info = mha_read_header(im_path);
im = mha_read_volume(info);

% final valid image
info_valdi = mha_read_header(valid_im_path);
im_val = mha_read_volume(info_valdi);

x = 0.5 ; % constant x for zy page (as shows)
y = 0.5 ; % constant y for xz page (as shows)
z = 0.5 ; % constant z for xy page (as shows)

% get certen image from 3d model
g = crap_3d_RT (im_path ,x , y , z);

g2 = crap_3d_RT (valid_im_path ,x , y , z);

s1 = g.s1; % Slice 1 (xy)
s2 = g.s2; % Slice 2 (yz)
s3 = g.s3; % Slice 3 (xz)

%name = [num2str(i) 'image.tif'];
imwrite(s1,'000SlICE1_res.tif');
% reaqd image as local var
im1 = imread('000SlICE1_res.tif');

%name = [num2str(i) 'image.tif'];
imwrite(g2.s1,'.\res\valid_res.tif');
% reaqd image as local var
valid_image = imread('.\res\valid_res.tif');


%% k-medious 3
clc;
close();
k = 3;
im1= im2double(im1);
[with , height] = size(im1);
im_res = zeros(with,height);
color_image = zeros(with, height, 3, 'uint8'); % automatic black! [0 is black] 3 = dimentions {R G B}

% using linier kmodues [1d]
d1_im = im1(:); %convert image to 1d [with * height * 1double]
d1_res = im_res(:);


imshow(im1); % shwo given image ...

opts = statset('Display','iter');
[idx,C,sumd,d,midx,info] = kmedoids(d1_im,k);


clusters = reshape(idx,with,[]);
for i= 1 : with
   for j = 1 : height
      if clusters(i , j) == 1 
          color_image(i,j , 3) = 255; %blue for k==1
      elseif clusters(i , j) == 2
          color_image(i,j , 1) = 255; %red for k==2
      elseif clusters(i , j) == 3
          color_image(i,j , 2) = 255; %green for k==3
      end
   end
end

file_name = '.\res\kmedoids3.tif';
imwrite(color_image,file_name);
imshow(color_image);



%% build mask from k
clc();
close();

masks.k1 = zeros(with , height);
masks.k2 = zeros(with , height);
masks.k3 = zeros(with , height);


for i= 1 : with
   for j = 1 : height
      if clusters(i , j) == 1 
          masks.k1(i,j) = 1;          %exists in K1
      elseif clusters(i , j) == 2
         masks.k2(i,j) = 1;  %exists in K2
      elseif clusters(i , j) == 3
          masks.k3(i,j) = 1; %exists in K3
      end
   end
end

% conver mask to binery [0 - 1] {redy for multiplex)
masks.k1 = imbinarize(masks.k1);
masks.k2 = imbinarize(masks.k2);
masks.k3 = imbinarize(masks.k3);

% show final image masked
subplot(1,3,1) ,imshow(im1 .* masks.k1) , title("MASKED 1") 
subplot(1,3,2) ,imshow(im1 .* masks.k2) , title("MASKED 2")
subplot(1,3,3) ,imshow(im1 .* masks.k3) , title("MASKED 3")


% saving results
file_name = '.\res\masked1.tif';
imwrite(im1 .* masks.k1,file_name);

file_name = '.\res\masked2.tif';
imwrite(im1 .* masks.k2,file_name);

file_name = '.\res\masked3.tif';
imwrite(im1 .* masks.k3,file_name);

%% compare with read valid image
clc();
close();
subplot(1 ,2 , 1)  , imshow (imread('.\res\masked3.tif')), title("our image");
subplot(1 ,2 , 2)  , imshow (imread('.\res\valid_res.tif')), title("ORGINAL SECTION");

%% Feature Extraction
clc();
close();
masked_im = imread('.\res\masked3.tif');
% hog ------------ [histogram of gradient]
feature = hog_feature_vector(masked_im); %hog features

% gabor ---------------
wavelength = 4; %just some numbers! [original matlab exam]
orientation = 90;
[mag,phase] = imgaborfilt(masked_im,wavelength,orientation);

%% train ....


