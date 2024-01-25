%% plot color pics
%% 导入颜色图像
clear; clc;

% 加载MAT文件
load('simulation_results\results\ARAD_1K_0314.mat');

% 指定保存路径
save_file = 'simulation_results\rgb_results\mst_s1\';
mkdir(save_file);

% 关闭所有图形窗口
close all;

% 设置帧数
frame = 1;

% 获取立方体数据
recon = cube;
intensity = 5;

% 循环处理每个通道
for channel = 1:31
    img_nb = channel;  % 通道号
    row_num = 1;
    col_num = 1;

    % 波段的波长
    lam31 = [400 410 420 430 440 450 460 470 480 490 500 510 ...
            520 530 540 550 560 570 580 590 600 610 620 630 ...
            640 650 660 670 680 690 700];

    % 将数据限制在 [0, 1] 范围内
    recon(find(recon > 1)) = 1;

    % 生成文件名
    name = [save_file 'frame' num2str(frame) 'channel' num2str(channel)];

    % 调用 dispCubeAshwin 函数显示图像并保存
    dispCubeAshwin(recon(:,:,img_nb), intensity, lam31(img_nb), [] ,col_num, row_num, 0, 1, name);
end

% 保持图形窗口打开
hold on;

