
I = imread('coins.png'); %Read grayscale image into the workspace.
BW = imbinarize(I);  %Convert the image into a binary image.
figure; imshowpair(I,BW,'montage') %Display the original image next to the binary version.
SE = strel ('disk', 5); %Create a disk-shaped structuring element with a radius of 5.
BW1 = imdilate(BW,SE); figure ; imshow(BW1) %Dilate the image 
SE1 = strel ('disk', 18); %Create a disk-shaped structuring element with a radius of 18.
BW2 = imerode(BW1,SE1); figure ; imshow(BW2) %Erode the image
SE2 = strel ('disk', 10); %Create a disk-shaped structuring element with a radius of 10.
BW3 = imdilate(BW2,SE2); figure ; imshow(BW3) %Dilate the image
