>> load('trainpo.mat'); 
>> gTruthFace = selectLabels(gTruth, 'face');
positiveInstances = objectDetectorTrainingData(gTruthFace);
>> gTruthFace = selectLabels(gTruth, 'face');
positiveInstances = objectDetectorTrainingData(gTruthFace);
>> %% Thư mục negative đã tạo
negativeFolder = 'D:\HK251\TGM\HW4\N';


%% Huấn luyện cascade detector
trainCascadeObjectDetector('faceDetector.xml', ...
    positiveInstances, negativeFolder, ...
    'NumCascadeStages', 20, ...
    'FeatureType', 'LBP', ...
    'ObjectTrainingSize', [24 24], ...
    'FalseAlarmRate', 0.1, ...
'TruePositiveRate', 0.995);



detector = vision.CascadeObjectDetector('faceDetector.xml');
detector.MergeThreshold = 6;    % giảm khung chồng nhau
detector.ScaleFactor = 1.05;     % dò tìm nhiều kích cỡ khuôn mặt

%% === 2. Đọc ảnh kiểm tra ===
img = imread('test2.jpg');

%% === 3. Phát hiện khuôn mặt ===
bbox = step(detector, img);

%% === 4. Gộp khung trùng nhau ===
minSize = 30;   % pixel
bbox = bbox(bbox(:,3) > minSize & bbox(:,4) > minSize, :);
%% === 5. Vẽ khung ===
detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, 'Face');
imshow(detectedImg);
title(sprintf('Detected %d face(s)', size(bbox,1)));