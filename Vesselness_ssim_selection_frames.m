% The code we suggest includes the vesselness method to enhance images and then we calculate the hyth
% threshold for each frame and we apply a function=Number of pixels/ Number
% of connected components. After that we choose the picks of func and among
% thme we apply ssim for the frame extraction

main_file_path ="C:\Coronary_sync";
main_file_list = dir(main_file_path);
main_file_list = main_file_list(~ismember({main_file_list.name}, {'.', '..'}));

option=2;

output_folder = fullfile("C:\Users\pyram\OneDrive\Thesis\Selection_of_Frames_Vesselness");
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

output_excel_path = "C:\Users\pyram\OneDrive\Thesis\Veselness_Hyth_ssim_results.xlsx";

if isfile(output_excel_path)
    existing_data = readcell(output_excel_path);  % Read the Excel file
    last_row = size(existing_data, 1) + 1;       % Find the last used row
else
    % Define headers if the file doesn't exist
    headers = {"Year","MainFileName", "DicomFileName", "Frame_Chosen", "Valid_NonValid","Reason"};
    existing_data = headers;
    last_row = 2;  % Start after the headers
end


output_data = {}; % Store new rows
row = 1;


for k=1:length(main_file_list)

    sub_filename=main_file_list(k).name;
    sub_fullpath=fullfile(main_file_path, sub_filename);
    year=extractBefore(sub_filename,"_");

    if isfolder(sub_fullpath)

        sub_filename_list=dir(sub_fullpath);
        sub_filename_list = sub_filename_list(~ismember({sub_filename_list.name}, {'.', '..'}));

        for f=1:length(sub_filename_list)
            patients_name=sub_filename_list(f).name;
            code = regexp(patients_name, '^\d+', 'match', 'once');
            ID_number = string(code);
            dicom_video_path=fullfile(sub_fullpath, patients_name, "IMAGE");

            dicom_files = dir(fullfile(dicom_video_path, '*'));
            dicom_files = dicom_files(~ismember({dicom_files.name}, {'.', '..'}));

            for d = 1:length(dicom_files)
                name_of_dicom_file=dicom_files(d).name;
                full_path_of_dicom=fullfile(dicom_video_path, name_of_dicom_file);

                if dicom_files(d).isdir
                    continue;
                end

                if isdicom(full_path_of_dicom)
                    image_data=dicomread(full_path_of_dicom);
                    image_info = dicominfo(full_path_of_dicom);
                    num_of_frames = size(image_data,4);


                    if num_of_frames==1
                        continue;
                    end

                    if num_of_frames<10
                        continue;
                    end

                    %The variables we will use
                    vesselCounts_hyth_ves=zeros(num_of_frames,1); %count of connected componets of hythresh by vesselness
                    vesselCounts_hyth_botom=zeros(num_of_frames,1); %count of connected componets of hythresh by bottom hat
                    vesselCounts_steger=zeros(num_of_frames,1);%count of connected componets of steger

                    sum_hyth_botom=zeros(num_of_frames,1); %sum of white pixels of hythresh by bottom hat
                    sum_hyth_ves=zeros(num_of_frames,1);%sum of white pixels of hythresh by vesselness
                    sum_steger=zeros(num_of_frames,1); %sum of white pixels of steger

                    %we find this by the connected components
                    sum_con1_ves=zeros(num_of_frames,1);%sum of pixels of hythresh by vesselness
                    sum_con2_ves=zeros(num_of_frames,1);%sum of pixels of hythresh by vesselness when sum >100
                    sum_con1_bot=zeros(num_of_frames,1);%sum of pixels of hythresh by bottom hat
                    sum_con2_bot=zeros(num_of_frames,1);%sum of pixels of hythresh by bottom hat when sum>100
                    sum_con1_steg=zeros(num_of_frames,1); %sum of pixels of steger
                    sum_con2_steg=zeros(num_of_frames,1);%sum of pixels of steger when sum>100

                    func1=zeros(num_of_frames,1);
                    func2=zeros(num_of_frames,1);
                    func3=zeros(num_of_frames,1);

                    acquis_time=image_info.AcquisitionTime;

                    for i=1:num_of_frames
                        testImage=image_data(:,:,:,i);

                        if size(testImage, 3) > 1
                            testImage= rgb2gray(testImage);  % Convert to grayscale if RGB
                        end

                        angles = 0:10:180-10;  % Directions for line SE
                        len = 30 ;                   % Length of structuring element
                        bottomHatResults = zeros(size(testImage, 1), size(testImage, 2), length(angles));

                        %Apply bottom-hat transform with linear SE at multiple angles
                        for j = 1:length(angles)
                            angle = angles(j);
                            se = strel('line', len, angle);  % Create line-shaped structuring element
                            bottomHatResults(:,:,j) = imbothat(testImage, se);
                        end

                        %Calculate the hythresh with botomHat of the image
                        %if option==1
                            botomhat = max(bottomHatResults, [], 3);
                            botomhat =255*mat2gray(botomhat); %normalize it to 0-256
                            hyth_botom = hysthresh(botomhat, 200,90);%apply thresold values
                            CC_hyth_botom=bwconncomp(hyth_botom);%calculate the connected components
                            vesselCounts_hyth_botom(i) = CC_hyth_botom.NumObjects;
                            region_b=regionprops(CC_hyth_botom);
                            squeezed_regionb=squeeze(cat(3,region_b.Area));
                            sum_con1_bot(i)=sum(squeezed_regionb);%sum of white pixels from all regions
                            sum_con2_bot(i)=sum(squeezed_regionb(squeezed_regionb>400));
                            sum_hyth_botom(i)=sum(hyth_botom(:));
                            func1(i)=sum_hyth_botom(i)/vesselCounts_hyth_botom(i);
                       % end

                        %if option==2
                            %Calculate the hythresh with vesselness of the image
                            vessel_image=vesselness2D(testImage, 2:5, [1;1], 1.5, false);
                            scaledImage = 255 * vessel_image; %normalize it to 0-256
                            hyth_ves = hysthresh(scaledImage, 230,140);%apply thresold values

                            hyth_ves(1:5, :)=0;
                            hyth_ves(end-5:end, :)=0;
                            hyth_ves(:, 1:5)=0;
                            hyth_ves(:, end-5:end)=0;

                            CC_hyth_ves=bwconncomp(hyth_ves);%calculate the connected components
                            region=regionprops(CC_hyth_ves);
                            vesselCounts_hyth_ves(i) = CC_hyth_ves.NumObjects;
                            squeezed_region=squeeze(cat(3,region.Area));
                            sum_con1_ves(i)=sum(squeezed_region);%sum of white pixels from all regions
                            sum_con2_ves(i)=sum(squeezed_region((squeezed_region>400)));%&(squeezed_region<1200)));                        
                            sum_hyth_ves(i)=sum(hyth_ves(:));
                            sum_hyth_ves(i)=sum(hyth_ves(:));
                            func2(i)=sum_con2_ves(i)/vesselCounts_hyth_ves(i);
                        %end

                        
                    %Calculate the steger image
                       % if option==3
                            sigmas=[2 4 6 8];
                            BW_all = false(size(testImage));  
                            for s=1:length(sigmas)

                                [BW,lmax,lmin,Hdet,Htrace,ux,uy] = steger_lines(testImage, 8, 0, 2);%calculate the BW with steger for more than one s
                                if (k==1)
                                    BW_all=BW;
                                else
                                    BW_all=BW_all|BW;%do or between images so we can keep more information
                                end
                            end

                            CC_steger=bwconncomp(BW_all);%calculate the connected components
                            vesselCounts_steger(i) = CC_steger.NumObjects;
                            region_s=regionprops(CC_steger);
                            squeezed_region_s=squeeze(cat(3,region_s.Area));
                            sum_con1_steg(i)=sum(squeezed_region_s);%suΦm of white pixels from all regions
                            sum_con2_steg(i)=sum(squeezed_region_s(squeezed_region_s>400));
                            sum_steger(i)=sum(BW_all(:));
                            func3(i)=sum_steger(i)/vesselCounts_steger(i);
                       % end

                        %vessel_image=uint16(vessel_image);
                        final_results=[vessel_image, hyth_ves; hyth_botom, BW_all];
                        final_results = insertText(final_results, [20, 20], "Vessel_Image", 'FontSize', 30, 'BoxColor', 'blue', 'TextColor', 'white');
                        final_results= insertText(final_results, [800, 20], "Hyth_Vess", 'FontSize', 30, 'BoxColor', 'blue', 'TextColor', 'white');
                        final_results= insertText(final_results, [20, 900], "Hyth_Botom", 'FontSize', 30, 'BoxColor', 'blue', 'TextColor', 'white');
                        final_results= insertText(final_results, [900, 900], "Steger", 'FontSize', 30, 'BoxColor', 'blue', 'TextColor', 'white');

                        figure('Visible','off');imshow(final_results,[]);
                        image_fig = sprintf('%s_%s_%s_Frame_%d.png', year, ID_number, name_of_dicom_file, i);
                        output_filename= fullfile(output_folder, image_fig);
                        imwrite(final_results, output_filename);

                        % fig1=figure;
                        % imshow(testImage,[]); title('Original');
                        % fig2=figure;
                        % imshow(vessel_image,[]); title('Vessel');
                        % fig3=figure;
                        % imshow(hyth_botom, []); title('Hyth_botom')
                        % fig4=figure;
                        % imshow(hyth_ves,[]); title('Hyth_vessel');
                        % fig5=figure;
                        % imshow(BW_all,[]); title('Steger');
                        % 
                        % subplot(2,2,1);

                        ...image_fig1 = sprintf('2021_11_2RS9TTKD_Frame_%d_og.png', i);
                        ...output_filename_fig1= fullfile(output_folder, image_fig1);
                        ...exportgraphics(fig1, output_filename_fig1);

                        ...image_fig2 = sprintf('2021_11_2RS9TTKD_Frame_%d_vessel.png', i);
                        ...output_filename_fig2= fullfile(output_folder, image_fig2);
                        ...exportgraphics(fig2, output_filename_fig2);

                        % image_fig3 = sprintf('2021_11_2RS9TTKD_Frame_%d_HythBot.png', i);
                        % output_filename_fig3= fullfile(output_folder, image_fig3);
                        % exportgraphics(fig3, output_filename_fig3);
                        
                        % image_fig4 = sprintf('2021_11_2RS9TTKD_Frame_%d_HythVes.png', i);
                        % output_filename_fig4= fullfile(output_folder, image_fig4);
                        % exportgraphics(fig4, output_filename_fig4);
                        % 
                        % image_fig5 = sprintf('2021_11_2RS9TTKD_Frame_%d_Steger.png', i);
                        % output_filename_fig5= fullfile(output_folder, image_fig5);
                        % exportgraphics(fig5, output_filename_fig5);
                    end

                    [pks,locs] = findpeaks(func2);
                    if isempty(pks)
                        continue
                    end

                    ssim_val=zeros(1, length(locs));
                    reference_image=image_data(:,:,:,1);
                    for p=1:length(locs)
                        current_image =image_data(:, :, :, locs(p));
                        ssim_val(p)=ssim(current_image, reference_image);
                    end

                    [value, idx]=min(ssim_val(:));
                    picked_frame=image_data(:,:,:,locs(idx));
                    fig_p=figure('Visible','off');
                    imshow(picked_frame,[]);
                    pick_f = sprintf('%s_%s_%s_%s_Selected_Frame%d_vessel.png', year, ID_number,acquis_time, name_of_dicom_file,locs(idx));
                    output_filename_fig= fullfile(output_folder, pick_f);
                    exportgraphics(fig_p, output_filename_fig);

                    %%Find peaks of botom hat method
                    [pks1,locs1] = findpeaks(func1);
                    if isempty(pks1)
                        continue
                    end

                    ssim_val_botom=zeros(1, length(locs1));
                    for b=1:length(locs1)
                        current_image_b =image_data(:, :, :, locs1(b));
                        ssim_val_botom(b)=ssim(current_image_b, reference_image);
                    end

                    [value1, idx1]=min(ssim_val_botom(:));
                    picked_frame_botom=image_data(:,:,:,locs1(idx1));
                    fig_botom=figure('Visible','off');
                    imshow(picked_frame_botom,[]);
                    pick_fb = sprintf('%s_%s_%s_%s_Selected_Frame%d_botom.png', year, ID_number,acquis_time, name_of_dicom_file,locs1(idx1));
                    output_filename_fig_b= fullfile(output_folder, pick_fb);
                    exportgraphics(fig_botom, output_filename_fig_b);

                    %%Find peaks of steger method
                    [pks3,locs3] = findpeaks(func3);
                    if isempty(pks3)
                        continue
                    end

                    ssim_val_steger=zeros(1, length(locs3));
                    for s=1:length(locs3)
                        current_image_s =image_data(:, :, :, locs3(s));
                        ssim_val_steger(s)=ssim(current_image_s, reference_image);
                    end

                    [value3, idx3]=min(ssim_val_steger);
                    picked_frame_steger=image_data(:,:,:,locs3(idx3));
                    fig_steger=figure('Visible','off');
                    imshow(picked_frame_steger,[]);
                    pick_fs = sprintf('%s_%s_%s_%s_Selected_Frame%d_steger.png', year, ID_number,acquis_time, name_of_dicom_file,locs3(idx3));
                    output_filename_fig_s= fullfile(output_folder, pick_fs);
                    exportgraphics(fig_steger, output_filename_fig_s);

                   all_images = zeros(size(image_data,1), size(image_data,2), 3, num_of_frames, 'uint8');

                    for frame=1:num_of_frames
                        testImage=image_data(:,:,:,frame);
                        testImage=im2uint8(mat2gray(testImage));
                        if size(testImage,3) == 1
                            testImage = repmat(testImage, 1, 1, 3);
                        end
                       
                        boxColors = {};
               
                        if exist('idx1','var') && frame == locs1(idx1)
                            boxColors{end+1} = 'red'; 
                        end
                        if exist('idx','var')  && frame == locs(idx)
                            boxColors{end+1} = 'green';
                        end
                        if exist('idx3','var') && frame == locs3(idx3)
                            boxColors{end+1} = 'blue'; 
                        end

                        
                        for l=1:numel(boxColors)
                            edge=(l-1)*6;
                            testImage = insertShape(testImage,'rectangle', [1+edge, 1+edge, 510-2*edge, 510-2*edge],'Color', boxColors{l}, 'LineWidth', 6);
                        end
                         all_images(:,:,:,frame) = testImage;
                    end

                    
                    rows = ceil(sqrt(num_of_frames));
                    cols = ceil(num_of_frames / rows);

                    figure('Visible','off');
                    montaged_image=montage(all_images, 'Size', [rows, cols]);
                    image_montaged = sprintf('%s_%s_%s.png', year, ID_number, name_of_dicom_file);
                    output_filename_montaged = fullfile(output_folder, image_montaged);
                    montaged_image = montaged_image.CData;
                    imwrite(montaged_image, output_filename_montaged);

                    % output_data{row,1}="2022";
                    % output_data{row,2}=patients_name;
                    % output_data{row,3}=name_of_dicom_file;
                    % output_data{row,4}=locs(idx);
                    % output_data{row,5}=".";
                    % output_data{row,6}=".";
                    % row = row + 1;

                    % cc_median = median(cc_list(k, f, :));
                    % for n = 1:num_of_frames
                    %     num_cc = cc_list(k, f, n);
                    % 
                    %     if num_cc > 2 * cc_median
                    %         continue;  % Skip noisy frames
                    %     end
                    % end

                    % s=fft(sum_hyth_ves(:));
                    % fig15=figure;
                    % plot(linspace(-7.5, 7.5, length(s)),abs(fftshift(s(:))));

                    fig12=figure('Visible', 'off');
                    plot(sum_con1_bot(:), 'r'); hold on; plot(sum_con2_bot(:), 'g'); title("Connected Components in Bottom-Hat Transformation:");
                    legend({'Sum CC pixels (all regions)', 'Sum CC pixels (>400)'},'Location','northeast','FontSize',10,'Box','on');hold off;
                    image_fig12 = sprintf('%s_%s_%s_Plot_sum_CC_pixels_bottom.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig12= fullfile(output_folder, image_fig12);
                    exportgraphics(fig12, output_filename_fig12);

                    fig12a=figure('Visible', 'off');
                    plot(sum_hyth_botom(:), 'b');hold on; title("Sum of pixels in Bottom-Hat Transformation:");hold off;
                    image_fig12a = sprintf('%s_%s_%s_Plot_sum_of_pixels_bottom.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig12a= fullfile(output_folder, image_fig12a);
                    exportgraphics(fig12a, output_filename_fig12a);
                    
                    fig13=figure('Visible', 'off');
                    plot(sum_con1_ves(:), 'r'); hold on; plot(sum_con2_ves(:), 'g'); title("Connected Components in Vesselness Transformation");
                    legend({'Sum CC pixels (all regions)', 'Sum CC pixels (>400)'},'Location','northeast','FontSize',10,'Box','on');hold off;
                    image_fig13 = sprintf('%s_%s_%s_Plot_sum_CC_pixels_ves.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig13= fullfile(output_folder, image_fig13);
                    exportgraphics(fig13, output_filename_fig13);

                    fig13a=figure('Visible', 'off');
                    plot(sum_hyth_ves(:), 'b');hold on; title("Sum of pixels in Vesselness Transformation:");hold off;
                    image_fig13a = sprintf('%s_%s_%s_Plot_sum_of_pixels_vesselness.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig13a= fullfile(output_folder, image_fig13a);
                    exportgraphics(fig13a, output_filename_fig13a);
                    
                    fig14=figure('Visible', 'off');
                    plot(sum_con1_steg(:), 'r'); hold on; plot(sum_con2_steg(:), 'g'); title("Connected Components in Steger Transformation"); 
                    legend({'Sum CC pixels (all regions)', 'Sum CC pixels (>400)'},'Location','northeast','FontSize',10,'Box','on');hold off;
                    image_fig14 = sprintf('%s_%s_%s_Plot_sum_CC_pixels_steger.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig14= fullfile(output_folder, image_fig14);
                    exportgraphics(fig14, output_filename_fig14);

                    fig14a=figure('Visible', 'off');
                    plot(sum_steger(:), 'b');hold on; title("Sum of pixels in Steger Transformation:");hold off;
                    image_fig14a = sprintf('%s_%s_%s_Plot_sum_of_pixels_steger.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig14a= fullfile(output_folder, image_fig14a);
                    exportgraphics(fig14a, output_filename_fig14a);

                    fig15=figure('Visible', 'off');
                    plot(func1(:)); hold on; plot(locs1, pks1, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r');title(["Frame score in Botom-Hat Transformation", num2str(max(func1(:)))]); hold off;
                    image_fig15= sprintf('%s_%s_%s_Plot_FrameScore_BotomHat.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig15= fullfile(output_folder, image_fig15);
                    exportgraphics(fig15,  output_filename_fig15);

                    fig16=figure('Visible', 'off');
                    plot(func2(:)); hold on; plot(locs, pks, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r'); title(["Frame score in Vesselness Transformation", num2str(max(func2(:)))]); hold off;
                    image_fig16= sprintf('%s_%s_%s_Plot_FrameScore_Vesselness.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig16= fullfile(output_folder, image_fig16);
                    exportgraphics(fig16,  output_filename_fig16);

                    fig17=figure('Visible', 'off');
                    plot(func3(:)); hold on; plot(locs3, pks3, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r'); title(["Frame score in Steger Transformation", num2str(max(func3(:)))]); hold off;
                    image_fig17= sprintf('%s_%s_%s_Plot_FrameScore_Steger.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig17= fullfile(output_folder, image_fig17);
                    exportgraphics(fig17,  output_filename_fig17);

                    fig18=figure('Visible', 'off');
                    plot(ssim_val_botom(:)); hold on; plot(idx1, value1, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r'); title("SSIM in Botom-Hat Transformation");hold off;
                    image_fig18 = sprintf('%s_%s_%s_Plot_SSIM_botomhat.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig18= fullfile(output_folder, image_fig18);
                    exportgraphics(fig18, output_filename_fig18);

                    fig19=figure('Visible', 'off');
                    plot(ssim_val(:)); hold on; plot(idx, value, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r'); title("SSIM in Vesselness Transformation");hold off;
                    image_fig19 = sprintf('%s_%s_%s_Plot_SSIM_vesselness.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig19= fullfile(output_folder, image_fig19);
                    exportgraphics(fig19, output_filename_fig19);

                    fig20=figure('Visible', 'off');
                    plot(ssim_val_steger(:)); hold on; plot(idx3, value3, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r'); title("SSIM in Steger Transformation");hold off;
                    image_fig20 = sprintf('%s_%s_%s_Plot_SSIM_steger.png',year, ID_number, name_of_dicom_file);
                    output_filename_fig20= fullfile(output_folder, image_fig20);
                    exportgraphics(fig20, output_filename_fig20);
                    
                    % 
                    % 
                    % [value_ssim, frame_ssim,ssim_all]=keyframe_selection_ssim(full_path_of_dicom);
                    % if frame_ssim>=num_of_frames-1 
                    %     max_frame=frame_ssim;
                    %     min_frame=frame_ssim-4;
                    %     if min_frame<1
                    %         min_frame=1;
                    %     end
                    % else
                    %     max_frame=frame_ssim+2;
                    %     min_frame=frame_ssim-2;
                    %     if min_frame<1
                    %         min_frame=1;
                    %     end
                    % end
                    % % 
                    % % if (locs(idx)>=min_frame) && (locs(idx)<=max_frame)
                    % %     count=count+1;
                    % %     %put the returned frame is right
                    % % end
                    % 
                    % 
                    % 
                    % for fr=min_frame:max_frame
                    %     formatted_image_name_ssim = sprintf('%s_%s_%s_%s_Frame%d.png', year, ID_number, acquis_time, name_of_dicom_file, fr);
                    %     output_filename_ssim = fullfile(output_folder, formatted_image_name_ssim);
                    %     testImage_ssim=image_data(:,:,:,fr);
                    %     testImage_ssim = uint8(255 * mat2gray(testImage_ssim));
                    %     imwrite(testImage_ssim, output_filename_ssim);
                    % end

                end
            end
        end
    end
end

% if ~isempty(output_data)
%     output_data_excel = [existing_data; output_data];  % Add new rows after existing data
%     writecell(output_data_excel, output_excel_path);

% end
