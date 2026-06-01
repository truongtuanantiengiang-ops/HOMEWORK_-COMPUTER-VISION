function [BW, T] = BasicGlobalThreshold(I, tol)
    % BASIC GLOBAL THRESHOLDING (heuristic)
    % I   : ảnh grayscale (uint8/double)
    % tol : sai số hội tụ (ví dụ 0.5)
    % BW  : ảnh nhị phân kết quả
    % T   : ngưỡng cuối cùng

    if size(I,3) == 3
        I = rgb2gray(I); 
    end
    I = double(I);

    % Ngưỡng ban đầu: trung bình toàn ảnh
    T = mean(I(:));

    while true
        % Tách pixel theo ngưỡng
        G1 = I(I >= T);
        G2 = I(I <  T);

        % Nếu 1 nhóm rỗng thì dừng
        if isempty(G1) || isempty(G2)
            break;
        end

        % Trung bình 2 nhóm
        m1 = mean(G1);
        m2 = mean(G2);

        % Ngưỡng mới
        T_new = (m1 + m2)/2;

        % Kiểm tra hội tụ
        if abs(T - T_new) < tol
            T = T_new;
            break;
        end

        T = T_new;
    end

    % Tạo ảnh nhị phân kết quả
    BW = I >= T;

end