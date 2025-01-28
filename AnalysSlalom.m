clear; clc; close all

 videoFileName = "SlalomK.mp4";
video = VideoReader(videoFileName);
videoFrame = read(video, 1);

% figure; hold on
% h = imshow(videoFrame);
% rot90

% title("Rightclick on the color for tracking")
  %targetColor = impixel
    targetColor =  [217   107   143] ; 

 th = 20;

% Vi gör bilden suddig med meningen för att underlätta trackingen.
%%

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
%  [bbox(2),bbox(2),bbox(2)+bbox(4),bbox(2)+bbox(4)],'w','FaceColor','none','EdgeColor','b');
% 
% 
nFrames = video.NumFrames;
time = NaN(nFrames, 1);
P = NaN(nFrames,2); 
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

% save("dataSlalom", "P", "time")

% smuttning av video använd dataposition

load("dataSlalom.mat")

% videoFileName = "Slalom.mp4"; % fuckar bupp här TestVideo.MOV
% video = VideoReader(videoFileName);
% videoFrame = read(video, 1);

% figure; 
hImage = imshow(videoFrame); hold on;
hPath = plot(P(:,1), P(:,2), '.k-');
hMarkers = plot(P(1,1), P(1,2), 'bo','MarkerFaceColor','c');
hPath2 = plot(NaN, NaN, '.c-');

startTime = 0; %seconds
endTime = 21;
idx = (time > startTime) & (time < endTime);
frames = find(idx);
time2 = time(idx);
P2 = P(idx,:);

span = 8; method = 'rloess';
P2 = [smooth(P2(:,1),span,method), smooth(P2(:,2),span,method)];
span = 3; method = 'moving';

for k = 1:15
P2 = [smooth(P2(:,1),span,method), smooth(P2(:,2),span,method)];

end

save('smoothdataSlalom','P2', 'time')

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
% hCornerPoints = plot(NaN,NaN,'+k','MarkerFaceColor','k'); % hörnpunktersvarta kors
% 
% % % % % Välj punkter
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
% save("planarHomographyLabS","P","p")

%% 

load("planarHomographyLabS.mat")
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

save("TestVideoHomographyS","H")

% title("Chose a pixel to get the position i meters")
% [u,v] = ginput(1)
% % % u = (u(2)+u(1))/2;
% % % v = (v(2)+v(1))/2;
% 
% hSamplePoint = plot(u,v,'ok','MarkerFaceColor','k');
% X = H\[u;v;1]
% x = X(1)/X(3)
% y = X(2)/X(3)
% title("nice")
%% 

load('smoothdataSlalom.mat')

P2m = zeros(nFrames,2) ;
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

M2 = [time,P2m(:,1),P2m(:,2)] ; 

%%
%frames
n=nFrames ; 

%hastighet
%skapa en array:
dfdtC_V = zeros(n,2);

%forward DIFF t1 +t2
dfdtC_V(1,:)= (P2m(2,:)-P2m(1,:))/time(1) ; 

%central DIFF
%x(i+1) framtid, x(i-1) dåtid
%generaliserad tid, elegant!

for i = 2:n-1

    dt = time(i-1) ; % time(i+1) - 
    dfdtC_V(i,:) = (P2m(i+1,:) - P2m(i-1,:))/dt ; 

end

%Back DIFF
dfdtC_V(nFrames,:)= (P2m(n,:)-P2m(n-1,:))/dt ;


%%%%%%%%%%%%%%%%%%%!! Acceleratioon !!%%%%%%%%%%%%%%%%%%%%%
%skapa en array:
A_dfdtC = zeros(n,2);

%forward DIFF t1 +t2
A_dfdtC(1,:)= (dfdtC_V(2,:)-dfdtC_V(1,:))/time(1) ; 

%central DIFF
for i = 2:n-1

    dt = time(i+1) - time(i-1) ; 
    A_dfdtC(i,:) = (dfdtC_V(i+1,:) - dfdtC_V(i-1,:))/dt ; 

end 

%Back DIFF
A_dfdtC(nFrames,:)= (P2m(nFrames,:)-P2m(n-1,:))/dt;


deltaX = dfdtC_V(:,1)   ; 
deltaY = -dfdtC_V(:,2) ; 
deltaT = diff (M2(:, 1));
deltaT = [0;deltaT];

% Beräkna hastighetens storlek (v) i varje tidssteg
hastighet = sqrt(deltaX.^2 + deltaY.^2) ./ deltaT;

% Beräkna enhetsvektorer för riktningen
evS = [deltaX ./ hastighet, deltaY ./ hastighet];

% Slalom = zeros(nFrames,2) ;

d2x3 = -[A_dfdtC(:,1)] ; 
d2y3 = -[A_dfdtC(:,2)] ; 

eaccS = [-d2x3 ./ hastighet, d2y3 ./ hastighet];


%% Plottning
hqv = quiver(P2(1,1),P2(1,2),evS(1,1),evS(1,2),5000,LineWidth=5);
hqacc = quiver(P2(1,1),P2(1,2),eaccS(1,1),eaccS(1,2),1000,LineWidth=5) ;

for i = 1:length(frames)
    % hMarkers = plot(P(1,1), P(1,2), 'bo','MarkerFaceColor','c');
    videoFrame = read(video, frames(i));
    hImage.CData = videoFrame;
     set(hqv,'XData', P2(i,1), 'YData',P2(i,2))
     set(hqv,'UData', evS(i,1), 'VData',evS(i,2))
      set(hqacc,'XData', P2(i,1), 'YData',P2(i,2))
      set(hqacc,'UData', eaccS(i,1), 'VData',eaccS(i,2))
         drawnow
     
end