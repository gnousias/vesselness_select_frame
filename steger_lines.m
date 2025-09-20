function [BW,lmax,lmin,Hdet,Htrace,ux,uy] = steger_lines(I, sigma,fast,denom) 
% this function is optimal for GPU computing
% (no matrix indexing, only elementwiase matrix operations)
% 1st input shouls be gpuArray
% the calling function should gather(outputs)

% TH=0.5;  argument
% TL=0.1;  argument
% sigma=5; argument
%wsize=3;  argument
%ax=3*sigma;
%stepx=2*ax/(2*wsize+1);
%ay=3*sigma;
%stepy=2*ay/(2*wsize+1);
%[x,y]=meshgrid(-ax:stepx:ax,-ay:stepy:ay);

%-------- CORRECTIONS ---------
sigma2 = sigma * sigma;
wsize = ceil(4*sigma);
[x,y] = meshgrid(-wsize : wsize);
G = exp(-(x.*x + y.*y) / (2*sigma2)) / (2*pi*sigma2);
Gx = -(x/sigma2).*G*sigma2;
Gy = -(y/sigma2).*G*sigma2;
%Gxx=((x.*x-sigma^4)/sigma^2).*G; LATHOS
Gxx = sigma2 * (x.*x - sigma2) .* G / (sigma2 * sigma2);
%Gxy=((x.*y)/sigma^4).*G;
Gxy = sigma2 * (x .* y) .* G / (sigma2 * sigma2);
%Gyy=((y.*y-sigma^4)/sigma^2).*G; LATHOS
Gyy = sigma2 * (y.*y - sigma2) .* G / (sigma2 * sigma2);
%figure; surf(x,y,Gxy);

% I = imread('C006p0151d19970121r0edir00_cropped_green.bmp');
%I = imread('photo.bmp.bmp');
if fast~=0
    Ixx = conv2(double(I), Gxx, 'same');
    Ixy = conv2(double(I), Gxy, 'same');
    Iyy = conv2(double(I), Gyy, 'same');
    Ix  = conv2(double(I), Gx,  'same');
    Iy  = conv2(double(I), Gy,  'same');
else
    Ixx = imfilter(double(I), Gxx,'conv', 'same', 'replicate');
    Ixy = imfilter(double(I), Gxy,'conv', 'same', 'replicate');
    Iyy = imfilter(double(I), Gyy,'conv', 'same','replicate');
    Ix  = imfilter(double(I), Gx,'conv',  'same','replicate');
    Iy  = imfilter(double(I), Gy,'conv',  'same','replicate');
end


%% mallon lathos
% Calculate eigenvalues
% % b = -(Ixx + Iyy);
% % c = Ixx.*Iyy - Ixy.*Ixy;
% % a = 1;
% % q = -0.5 * (b + sign(b) .* real(sqrt(b.*b - 4*a*c)));
% % l1 = q/a;  % !!!! ./ ??
% % l2 = c./q;
% lmax = max(l1,l2);
% lmin = min(l1,l2);
%% replaced by

Hdet=Ixx.*Iyy-Ixy.^2;
Htrace=Ixx+Iyy;

riza=sqrt((Ixx-Iyy).^2+4*Ixy.^2);
l1=(Ixx+Iyy+riza)/2;
l2=(Ixx+Iyy-riza)/2;

[lmax,idmax] = max(cat(3,abs(l1),abs(l2)),[],3);
lmax(idmax==1)=l1(idmax==1);
lmax(idmax==2)=l2(idmax==2);

[lmin,idmin] = min(cat(3,abs(l1),abs(l2)),[],3);
lmin(idmin==1)=l1(idmin==1);
lmin(idmin==2)=l2(idmin==2);

% Select the max positive eigenvalue for dark vessels
% indx = find(lmax > 0);
% % Calclulate eigenvectors for the selected eigenvalues
% ux = 1 ./ sqrt(1 + ((lmax(indx) - Ixx(indx)).^2) ./ (Ixy(indx).^2));
% uy = ((lmax(indx)-Ixx(indx)).*ux)./Ixy(indx);

temp_mat=double(lmax > max(lmax)/denom);
ux = 1 ./ sqrt(1 + ((lmax.*temp_mat - Ixx.*temp_mat).^2) ./ ((Ixy.*temp_mat).^2));
uy = ((lmax.*temp_mat-Ixx.*temp_mat).*ux)./(Ixy.*temp_mat);
    
% Find the pixels with vanishing directional derivative
% t = -(ux.*Ix(indx) + uy.*Iy(indx)) ./ (ux.*ux.*Ixx(indx) + uy.*uy.*Iyy(indx) + 2*ux.*uy.*Ixy(indx));
% p = find(abs(t.*ux) + abs(t.*uy) < 1);
% indx1 = indx(p);
% BW = uint8(zeros(size(I)));
% BW(indx1) = 255;

% t = -(ux.*Ix.*temp_mat + uy.*Iy.*temp_mat) ./ (ux.*ux.*Ixx.*temp_mat + uy.*uy.*Iyy.*temp_mat + 2*ux.*uy.*Ixy.*temp_mat);
% p = double((abs(t.*ux) + abs(t.*uy) < 5) & (abs(t.*ux) + abs(t.*uy)>0));
% BW = double(zeros(size(I)));
% BW=BW+255*p.*temp_mat;
t = -(ux.*Ix.*temp_mat + uy.*Iy.*temp_mat) ./ (ux.*ux.*Ixx.*temp_mat + uy.*uy.*Iyy.*temp_mat + 2*ux.*uy.*Ixy.*temp_mat);
p = double((abs(t.*ux) + abs(t.*uy) < 1) & (abs(t.*ux) + abs(t.*uy)>0));
BW = double(zeros(size(lmax)));
BW=BW+255*p.*temp_mat;

    
% for i=1+wsize:size(Ixx,2)-wsize
%     
%     for j=1+wsize:size(Ixx,1)-wsize
%         H=[Ixx(j,i),Ixy(j,i);Iyx(j,i),Iyy(j,i)];
%         [V,D] = eig(H);
%         [m,imin]=min(sum(D));
%         u=V(:,imin);
%         t=-(u(1)*Ix(j,i)+u(2)*Iy(j,i))/(u(1)*u(1)*Ixx(j,i)+2*u(1)*u(2)*Ixy(j,i)+u(2)*u(2)*Iyy(j,i));
%         if sum(abs(t*u)<0.5)==2 & abs(D(imin,imin))>TH
%             I1(j,i)=255;
%         end
%     end
% end