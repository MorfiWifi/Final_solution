function out = crap_3d_RT (file_name , DXP , DYP , DZP )
info = mha_read_header(file_name);
im = mha_read_volume(info);
xyzs = size(im);
x = xyzs (1 ,1);
y = xyzs (1,2);
z = xyzs (1,3);

DX = x * DXP;
DY = y * DYP;
DZ = z * DZP;

DX = int32(DX);
DY = int32(DY);
DZ = int32(DZ);

S1 = zeros(x,y);
for i = 1 :1:x
    for j = 1 :1:y
    S1(i,j) = im(i,j,DZ);
    end
end

S2 = zeros(y,z);
for i = 1 :1:z
    for j = 1 :1:y
    S2(j,i) = im(DX,j,i);
    end
end

S3 = zeros(x,z);
for i = 1 :1:x
    for j = 1 :1:z
    S3(i,j) = im(i,DY,j);
    end
end

S1 = mat2gray(S1);
S2 = mat2gray(S2);
S3 = mat2gray(S3);


%subplot (2,2,1);imshow(S1);title('Slice 1');
%subplot (2,2,2);imshow(S2);title('Slice 2');
%subplot (2,2,3);imshow(S3);title('Slice 3');

out.s1 = S1;
out.s2 = S2;
out.s3 = S3;

end