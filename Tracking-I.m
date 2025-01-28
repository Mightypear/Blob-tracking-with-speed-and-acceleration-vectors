clear; clc; close all

% videoFileName = "TestVideo.MOV";
 videoFileName = "acc.mp4";
video = VideoReader(videoFileName);
videoFrame = read(video, 1);

% figure;
%h = imshow(videoFrame);
% rot90

% title("Rightclick on the color for tracking")
  % targetColor = impixel
 % targetColor = [249   151     7] % testvideo
 targetColor =  [  217    95    97] ;  % acc

 th = 50;

% Vi gör bilden suddig med meningen för att underlätta trackingen.
rgbImage = imgaussfilt(videoFrame,2); %Smoothing

% figure; imshow(rgbImage);
% title('Filtered Image');
% hold on
% plot(500, 500, 'o','markerFaceColor',targetColor/255,'markersize',20)

% Nu omvandlar vi bilden till en maskering så att den valda färgen är vit
% och allt annat svart. Det går att blob trackerns jobb mycket enklare och
% möjligt.
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

mask = (redChannel < targetColor(1)+th) & (redChannel > targetColor(1)-th) & ...
       (greenChannel < targetColor(2)+th) & (greenChannel > targetColor(2)-th) & ...
       (blueChannel < targetColor(3)+th) & (blueChannel > targetColor(3)-th);

% figure; h_mask = imshow(mask);
% title('Mask Image');

blob = vision.BlobAnalysis('MaximumCount', 1);
[blobArea, centroid, bbox] = blob(mask);

%  figure; %set(gcf,'Position',[2654 71 2086 1163]) figur 2?
%  hImage = imshow(videoFrame); hold on;
% hMarkers = plot(NaN, NaN, 'bo','MarkerFaceColor','y'); % punkten som följer bilen
% hPath = plot(NaN, NaN, '.k-');
% h_box = patch([bbox(1),bbox(1)+bbox(3),bbox(1)+bbox(3),bbox(1)], ...
% [bbox(2),bbox(2),bbox(2)+bbox(4),bbox(2)+bbox(4)],'w','FaceColor','none','EdgeColor','b');
% 
% 
% 
% nFrames = video.NumFrames;
% time = NaN(nFrames, 1);
% P = NaN(nFrames,2); 
% 
% 
% for i = 1:nFrames
%     videoFrame = read(video, i);
%     rgbImage = imgaussfilt(videoFrame,2); % Gör den suddig
%     redChannel = rgbImage(:, :, 1);
%     greenChannel = rgbImage(:, :, 2);
%     blueChannel = rgbImage(:, :, 3);
%     mask = (redChannel < targetColor(1)+th) & (redChannel > targetColor(1)-th) & ...
%            (greenChannel < targetColor(2)+th) & (greenChannel > targetColor(2)-th) & ...
%            (blueChannel < targetColor(3)+th) & (blueChannel > targetColor(3)-th);
% 
%     h_mask.CData = mask;
% 
%     [blobArea, centroid, bbox] = blob(mask);
% 
%     P(i,:) = centroid;
% 
%     dt = 1/video.FrameRate;
%     time(i) = dt * i;
% 
%     hImage.CData = videoFrame;
%     set(hMarkers,'XData', P(i,1), 'YData',P(i,2))
%     set(hPath,'XData', P(1:i,1), 'YData',P(1:i,2))
% 
%     title(num2str(time(i)))
%     drawnow
% end


% save("data", "P", "time")
%%

% smuttning av video använd dataposition

load("data.mat")

videoFileName = "acc.mp4"; % fuckar bupp här TestVideo.MOV
video = VideoReader(videoFileName);
videoFrame = read(video, 1);

% figure; 
hImage = imshow(videoFrame); hold on;
hPath = plot(P(:,1), P(:,2), '.k-');
hMarkers = plot(P(1,1), P(1,2), 'bo','MarkerFaceColor','c');
hPath2 = plot(NaN, NaN, '.c-');

startTime = 0; %seconds
endTime = 1.58;
idx = (time > startTime) & (time < endTime);
frames = find(idx);
time2 = time(idx);
P2 = P(idx,:);

span = 8; method = 'rloess';
P2 = [smooth(P2(:,1),span,method), smooth(P2(:,2),span,method)];
span = 3; method = 'moving';

for k = 1:93
P2 = [smooth(P2(:,1),span,method), smooth(P2(:,2),span,method)];

end

save('smoothdata','P2', 'time')
%%

for i = 1:length(frames)
    videoFrame = read(video, frames(i));
    hImage.CData = videoFrame;
    set(hMarkers, XData=P2(i,1), YData=P2(i,2))
    set(hPath2, XData=P2(1:i,1), YData=P2(1:i,2))
    drawnow
end

% figure; 
% hImage = imshow(videoFrame);
% hold on;
hCornerPoints = plot(NaN,NaN,'+k','MarkerFaceColor','k'); % hörnpunktersvarta kors

% Välj punkter
% x = NaN(4,1); y = x;
% for i = 1:4
% title(sprintf('Select point %d',i))
% [xi,yi] = ginput(1);
% x(i) = xi; y(i) = yi;
% set(hCornerPoints,'XData', x, 'YData', y);
% end
% title('All four points are chosen.')
% p = [x,y]
% 
% P = [0.0, 0.0
% 3.0, 0.0
% 3.0, 4.5
% 0.0, 4.5]
% save("planarHomographyLab","P","p")

load("planarHomographyLab.mat")
% set(hCornerPoints,'XData', p(:,1), 'YData', p(:,2));

x = P(:,1); y = P(:,2);
u = p(:,1); v = p(:,2);

A = zeros(9,9); j = 1;
for i = 1:4
A(j, :) = -[-x(i), -y(i), -1, 0, 0, 0, x(i)*u(i), y(i)*u(i), u(i)];
A(j+1,:) = -[ 0, 0, 0, -x(i), -y(i), -1, x(i)*v(i), y(i)*v(i), v(i)];
j = j + 2;
end
A(end,end) = 1;
b = zeros(9,1); b(end) = 1;
H = reshape(A\b, 3,3)';
% vpa(H,3)

save("TestVideoHomography","H")

% title("Chose a pixel to get the position i meters")
% [u,v] = ginput(1)
% u = (u(2)+u(1))/2;
% v = (v(2)+v(1))/2;

% hSamplePoint = plot(u,v,'ok','MarkerFaceColor','k');
% X = H\[u;v;1]
% x = X(1)/X(3)
% y = X(2)/X(3)
% title("nice")
%% 

load('smoothdata.mat')

P2m = zeros(93,2) ;
    for i = 1: size(P2, 1)
    % Hämta värden från den i:te raden
    U = P2(i, 1);  % Värde från första kolumnen
    V = P2(i, 2);  % Värde från andra kolumnen
    
    % Skapa en vektor [A; B; 1]
    v = [U; V; 1];
      
    X = H\v ;

    P2m(i,1) = X(1)/X(3) ;
    P2m(i,2) = X(2)/X(3) ; 
    
end

M = [time,P2m(:,1),P2m(:,2)] ; 


%%
% Anta att M är 93x3 matris
% M(:,1) = tid, M(:,2) = x-koordinat, M(:,3) = y-koordinat

% Beräkna skillnader i tid, x och y
deltaT = diff(M(:, 1));    % Tidsskillnad mellan successiva mätpunkter
deltaX = diff(M(:, 2));    % Skillnad i x-koordinat
deltaY = diff(M(:, 3));    % Skillnad i y-koordinat


% Beräkna hastighetens storlek (v) i varje tidssteg
hastighet = sqrt(deltaX.^2 + deltaY.^2) ./ deltaT;

% Beräkna enhetsvektorer för riktningen
env = [-deltaX ./ hastighet, -deltaY ./ hastighet];

% Visa hastighet och riktning som enhetsvektor

for i = 1:length(hastighet)
    fprintf('Vid tid %f s: Hastighet = %f, Enhetsvektor = [%f, %f]\n', ...
    M(i, 1), hastighet(i), env(i, 1), env(i, 2));
end
newrow = [0,0] ; 
env3 = [newrow;env] ; 

% acc
% deltaV = diff(env3(:,1))
% deltaV = diff(env3(:,2))
% deltaV = diff(env3(:,3))
d2x = diff(deltaX)  ; 
d2y = diff(deltaY) ; 
acc = zeros(93,2) ;

cero = [0] ; 
d2x3 = [cero;cero;d2x] ; 
d2y3 = [cero;cero;d2y] ; 
hastighet1 = [cero;hastighet] ;
% accel = sqrt(d2x3.^2 + d2y3.^2) ./ deltaT;
enacc = [-d2x3 ./ hastighet1, -d2y3 ./ hastighet1];
% acc = [d2x(:,1),d2y(:,2)]


%%
% figure ; hold on ; axis equal
% title ('hastighet')
% hImage3 = imshow(videoFrame); hold on;
hqv = quiver(P2(1,1),P2(1,2),env3(1,1),env3(1,2),10000,LineWidth=5);

hqacc = quiver(P2(1,1),P2(1,2),enacc(1,1),enacc(1,2),100000,LineWidth=5) ; 
% video = VideoReader(videoFileName);
% videoFrame = read(video, 1);

 %set(hqv,'XData',x,'Ydata',y,'UData',vx,'VData',vy)
 % figure

% video = VideoReader (videoFileName);
% videoFrame = read (video, frames (1));
% % figure; %set(gcf, 'Position', [-2368, 88, 2094, 1173])
% hImage = imshow(videoFrame); hold on;

for i = 1:length(frames)
    % hMarkers = plot(P(1,1), P(1,2), 'bo','MarkerFaceColor','c');
    videoFrame = read(video, frames(i));
    hImage.CData = videoFrame;
     set(hqv,'XData', P2(i,1), 'YData',P2(i,2))
     set(hqv,'UData', env3(i,1), 'VData',env3(i,2))
      set(hqacc,'XData', P2(i,1), 'YData',P2(i,2))
     set(hqacc,'UData', enacc(i,1), 'VData',enacc(i,2))
         drawnow
     
end


%% mirzaz kod
% video = VideoReader (videoFileName);
% videoFrame = read (video, frames (1));
% figure; %set(gcf, 'Position', [-2368, 88, 2094, 1173])
% hImage = imshow(videoFrame); hold on;
% hPath = plot (NaN, NaN, 'ok-', 'MarkerFaceColor', 'k');
% set(hPath, 'XData", P(:,1), 'YData', P(:,2));
% hSmoothPath = plot (P2(:,1), P2(:,2), 'oc-', 'MarkerFaceColor','c');
% hMarkers = plot (NaN, NaN, 'bo', 'MarkerFaceColor', 'y');
% h_v = quiver (NaN, NaN, NaN, NaN, 0.5, 'b', 'LineWidth', 2);
% h_a quiver (NaN, NaN, NaN, NaN, 0.5, 'r', 'LineWidth', 2);
% for i = 1:nFrames
% videoFrame = read(video, frames(i));
% hImage.CData = videoFrame;
% set(h_v, 'XData', P2(i, 1), 'YData', P2(1,2), 'Udata', v2 (i, 1), 'VData',v2(1,2))
% set(h_a, 'XData', P2(i, 1), 'YData', P2(i, 2), 'Udata', a2(1,1), 'VData', a2 (1,2))
% set (hMarkers, 'XData', P2(i, 1), 'YData', P2(1,2))
% drawnow
% pause(1/30)
% end

%%

% SmoothCarPositionData
video = VideoReader(videoFileName);
videoFrame = read(video,frames(1));
figure; %set(gcf,'Position',[2654 71 2086 1163])
hImage = imshow(videoFrame); hold on;
h_marker = plot(NaN, NaN, 'yo','MarkerFaceColor','y');
hPath = plot(NaN, NaN, '.k-','DisplayName','Raw Position Data (px)');
hSmoothPath = plot(NaN, NaN, '.c-','DisplayName','Smooth Position Data (px)');
legend show
video.CurrentTime = time(1);
nFrames = length(time);
title(['Analysis on averaged data'])
% exportgraphics(gcf,'FilteredData.gif','Append',false);
videoRecordingObj = VideoWriter(['FilteredData.mp4'], 'MPEG-4');
open(videoRecordingObj);
for j = 1:length(frames)
    % if j = 1
    % set(hPath,'XData', P(1:j,1),'YData', P(1:j,2));
    % end
    videoFrame = read(video,frames(j));
    set(hPath,'XData', P(1:frames(j),1),'YData', P(1:frames(j),2));
    set(hSmoothPath,'XData', P2(1:j,1),'YData', P2(1:j,2));
    set(h_marker,'XData', P2(j,1),'YData', P2(j,2));
    hImage.CData = videoFrame;
    drawnow
    pause(1/video.FrameRate)
    % set(h_v,'XData',P2(j,1),'YData',P2(j,2),'Udata',v2(j,1),'VData',v2(j,2))
    % set(h_a,'XData',P2(j,1),'YData',P2(j,2),'Udata',a2(j,1),'VData',a2(j,2))
    try
    frame = getframe(gcf);
    writeVideo(videoRecordingObj,frame);
    catch
    close(videoRecordingObj)
    end
    if mod(j,10)==0 % Spara var 10e frame
    % exportgraphics(gcf,'FilteredData.gif','Append',false);
    end
end
close(videoRecordingObj)
