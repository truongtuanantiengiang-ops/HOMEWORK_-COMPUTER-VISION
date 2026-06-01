RGB = imread('day3.jpg');           % ảnh đầu vào
I = im2gray(RGB);                   % chuyển ảnh về xám 
BW = edge(I, 'canny');              % phát hiện biên Canny
[H, theta, rho] = hough(BW);        % Biến đổi Hough
% --- Tìm đỉnh có giá trị mạnh nhất
peaks = houghpeaks(H, 1, 'Threshold', ceil(0.3 * max(H(:))));
% --- Dò các đường thẳng tương ứng với đỉnh ---
lines = houghlines(BW, theta, rho, peaks, 'FillGap', 5, 'MinLength', 20);
    % -Tính góc nghiêng của sợi dây
    angle = lines(1).theta + 90;         % Góc thực tế của sợi dây
    if abs(angle) < 10
        state = 'Dây nằm ngang';
    elseif abs(angle) > 80
        state = 'Dây đứng thẳng';
    else
        state = 'Dây nghiêng';
    end
    figure;
    imshow(RGB); hold on;
    % Vẽ đường thẳng phát hiện được
    xy = [lines(1).point1; lines(1).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'red');
    % Vẽ đầu mút
    plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
    plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'green');
    % Hiển thị thông tin
    title(['Trạng thái: ' state ', góc = ' num2str(angle) '°']);
    disp(['Trạng thái sợi dây: ' state]);
