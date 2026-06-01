folder_path = "C:\Users\Thiện Khả\Documents\HK_251\Thị giác máy\HW5\20picture";
image_files = dir(fullfile(folder_path, '*.jpg'));

faceDetector = vision.CascadeObjectDetector;


for imgIndex = 1:length(image_files)
    figure('Name', ['Face Detection - Image ' num2str(imgIndex)]); 


    img_path = fullfile(folder_path, image_files(imgIndex).name);
    image = imread(img_path);

    bbox = step(faceDetector, image);

    if ~isempty(bbox)
        image_face = insertObjectAnnotation(image, 'rectangle', bbox, 'Face');
    else
        image_face = image;
    end

         
    imshow(image_face);
    title(['Image ' num2str(imgIndex)], 'Interpreter', 'none');

end
