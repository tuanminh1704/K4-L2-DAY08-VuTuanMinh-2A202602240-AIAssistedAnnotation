# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Vũ Tuấn Minh

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian và có vùng đệm ở giữa thay vì chia ngẫu nhiên. Camera đứng một chỗ nên một chiếc xe có thể xuất hiện trong nhiều khung hình liên tiếp. Nếu chia ngẫu nhiên, các ảnh của cùng một chiếc xe hoặc cùng một cảnh có thể xuất hiện ở cả tập học và tập kiểm thử. Khi đó mô hình có thể nhìn thấy những cảnh rất giống nhau trong quá trình học và khi kiểm tra, làm cho điểm trên tập kiểm thử cao hơn khả năng tổng quát thực tế của mô hình.

## 2. Mô hình khởi đầu lạnh (cold start)

Ở vòng 0, mô hình `yolov8n cold start (COCO car+bus+truck)` được đánh giá trên tập kiểm thử gồm 20 ảnh và 403 box tham chiếu. Có 14 box cao dưới 16 px được bỏ qua. Với ngưỡng IoU 0.5 và conf 0.25, AP50 của vòng 0 là 0.771, Precision là 0.925, Recall là 0.489 và F1 là 0.640.

Recall theo kích thước xe lần lượt là 0.182 đối với xe nhỏ, 0.547 đối với xe vừa và 0.561 đối với xe lớn. Điều này cho thấy mô hình khởi đầu lạnh gặp khó khăn rõ rệt với các xe nhỏ ở xa. Khả năng tìm thấy xe vừa và xe lớn tốt hơn nhưng vẫn còn các trường hợp bỏ sót hoặc bounding box chưa khớp.

Qua `outputs/compare_round0.jpg`, có thể thấy một số bounding box của mô hình không khớp hoàn toàn với nhãn tham chiếu, đặc biệt ở các xe nhỏ, xe ở xa hoặc các trường hợp khó quan sát. Một trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai là xe rất nhỏ, bị che khuất hoặc nằm trong vùng sáng của ảnh. Nhãn tham chiếu ban đầu cũng được tạo bởi mô hình và chưa được người kiểm tra thủ công toàn bộ, nên bounding box tham chiếu có thể bị thiếu hoặc lệch. Vì vậy một điểm đánh giá thấp chưa chắc hoàn toàn do mô hình bị sai.

## 3. Chiến lược chọn mẫu

Điểm chọn mẫu được tính theo công thức:

`score = W_U·U + W_A·A + W_D·D`

Trong đó `U` biểu diễn độ bất định của mô hình, `A` biểu diễn mức độ các bounding box còn lưỡng lự và `D` biểu diễn mức độ khác biệt về thời gian so với các ảnh khác. Ba thành phần được kết hợp bằng các trọng số tương ứng để tạo ra điểm ưu tiên cho từng frame.

`MIN_GAP_S` dùng để đảm bảo hai frame được chọn phải cách nhau ít nhất một khoảng thời gian nhất định. Vì camera đứng yên nên các frame quá gần nhau thường gần như cùng một cảnh. Việc đặt khoảng cách tối thiểu giúp tránh dùng ngân sách rà nhãn cho nhiều ảnh gần như trùng nhau.

Nếu chỉ có ngân sách rà năm ảnh, tôi ưu tiên `frame_0297.jpg` với score 0.7974 tại thời điểm 118.8 giây, `frame_0018.jpg` với score 0.7151 tại 7.2 giây, `frame_0029.jpg` với score 0.7127 tại 11.6 giây, `frame_0074.jpg` với score 0.7086 tại 29.6 giây và `frame_0002.jpg` với score 0.6607 tại 0.8 giây.

Ba frame thuộc lô 12 ảnh được model chọn mà tôi dùng làm ví dụ là `frame_0297.jpg`, `frame_0018.jpg` và `frame_0029.jpg`. `frame_0297.jpg` có score 0.7974, gồm 10 box và 6 box chưa chắc chắn. `frame_0018.jpg` có score 0.7151, gồm 7 box và 4 box chưa chắc chắn. `frame_0029.jpg` có score 0.7127, gồm 10 box và 4 box chưa chắc chắn. Các thông tin này cho thấy những frame này có nhiều vùng cần con người kiểm tra.

Một frame có điểm cao nhưng không được ưu tiên là `frame_0295.jpg`, có score 0.7191 và đứng hạng 2. Tuy nhiên frame này ở thời điểm 118.0 giây, chỉ cách `frame_0297.jpg` 0.8 giây. `frame_0296.jpg` ở 118.4 giây cũng nằm giữa hai frame này. Vì vậy chọn `frame_0297.jpg` thay vì chọn cả các frame rất gần nhau giúp giảm công rà nhãn cho những ảnh gần như cùng một cảnh.

Điểm selection cao không chứng minh rằng sửa frame đó chắc chắn sẽ làm mô hình tốt hơn. Điểm cao chỉ cho biết frame có mức độ bất định hoặc tín hiệu cần xem xét cao hơn theo tiêu chí của bộ chọn. Hiệu quả thực tế phải được đánh giá bằng kết quả sau khi cập nhật nhãn và train lại trên cùng tập kiểm thử.

## 4. Các vòng học chủ động (active learning)

Kết quả trên tập kiểm thử 20 ảnh như sau:

| Vòng | Model | Ảnh train | Box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vòng 1..1 | 12 | 331 | 0.763 | -0.009 | 1.000 | 0.117 | 0.209 | 0.000 | 0.061 | 0.707 |

Ở vòng 1, có 12 ảnh được dùng để train. Model ban đầu đề xuất 169 box, sau khi rà và sửa còn 331 box. Có 44 box được giữ nguyên, 101 box được chỉnh sửa, 24 box bị xoá và 186 box được thêm mới. Tỷ lệ box được chấp nhận nguyên trạng chỉ là 26%. Điều này cho thấy pre-label ban đầu còn khá nhiều lỗi và cần sự can thiệp của người gán nhãn.

Sau fine-tune, AP50 giảm từ 0.771 xuống 0.763, tương ứng giảm 0.009 so với cold start. Precision tăng từ 0.925 lên 1.000, nhưng Recall giảm mạnh từ 0.489 xuống 0.117 và F1 giảm từ 0.640 xuống 0.209.

Theo kích thước xe, recall xe nhỏ giảm từ 0.182 xuống 0.000, recall xe vừa giảm từ 0.547 xuống 0.061, trong khi recall xe lớn tăng từ 0.561 lên 0.707. Như vậy vòng 1 có sự cải thiện ở nhóm xe lớn nhưng làm giảm mạnh khả năng phát hiện xe nhỏ và xe vừa.

Từ `round1_diff.md`, số lượng thay đổi là 44 box giữ nguyên, 101 box chỉnh sửa, 24 box xoá và 186 box thêm mới. Số box thêm mới rất lớn so với số box được giữ nguyên, cho thấy pre-label ban đầu đã bỏ sót nhiều đối tượng. Đặc biệt, tổng số box sau khi sửa là 331, gần gấp đôi số 169 box mà model ban đầu đề xuất.

Trong `REVIEW_LOG.csv`, có một số trường hợp cụ thể. Ở `frame_0187.jpg`, một xe nhỏ ở phía xa được thêm bounding box vì AI bỏ sót xe. Ở `frame_0270.jpg`, bounding box của xe sát mép ảnh được chỉnh lại để ôm sát phần thân xe thực tế nhìn thấy. Ở `frame_0331.jpg`, bounding box trên vùng sáng mặt đường được xoá vì đó là ánh sáng phản chiếu chứ không phải xe.

`BLIND_SCAN.md` thể hiện những gì tôi quan sát độc lập trước khi xem hoặc chỉnh kết quả model. `REVIEW_LOG.csv` ghi lại những lỗi pre-label cụ thể mà tôi đã sửa, còn `round1_diff.md` tổng hợp toàn bộ thay đổi giữa box model đề xuất và box sau khi rà. Các chỉ số trong `rounds_table.md` lại phản ánh kết quả của model sau khi fine-tune. Vì vậy cần phân biệt quan sát của người gán nhãn, lỗi pre-label đã sửa và kết quả của model sau khi học lại.

Khi so sánh `compare_round0.jpg` và `compare_round1.jpg`, cần chú ý các trường hợp bounding box thay đổi sau fine-tune. Số liệu cho thấy model sau fine-tune có Precision 1.000 nhưng Recall chỉ còn 0.117. Điều này phù hợp với việc model trở nên thận trọng hơn: các prediction còn lại có độ chính xác cao nhưng nhiều xe bị bỏ sót. Trường hợp cần đặc biệt kiểm tra là các xe nhỏ và vừa, vì recall của hai nhóm này giảm rất mạnh sau fine-tune.

Một ca khó theo guideline là xe nhỏ ở xa. Khi xe chỉ chiếm một vùng rất nhỏ trong ảnh, bounding box khó xác định chính xác và dễ nhầm với ánh sáng hoặc vật thể khác. Đây cũng là nhóm mà recall giảm xuống 0.000 ở vòng 1, nên cần được ưu tiên rà lại trước khi thực hiện thêm một vòng train.

## 5. Kết luận và giới hạn

Kết quả vòng 1 chưa cho thấy sự cải thiện so với cold start. AP50 giảm từ 0.771 xuống 0.763, tương ứng giảm 0.009. Precision tăng từ 0.925 lên 1.000 nhưng Recall giảm từ 0.489 xuống 0.117 và F1 giảm từ 0.640 xuống 0.209.

Vì AP50 và F1 đều giảm, tôi chọn dừng ở vòng này để kiểm tra lại nhãn và nguyên nhân của việc giảm Recall trước khi train thêm. Đặc biệt, recall của xe nhỏ giảm từ 0.182 xuống 0.000 và xe vừa giảm từ 0.547 xuống 0.061. Điều này cho thấy việc train thêm ngay có thể tiếp tục làm mô hình bỏ sót các nhóm xe này nếu nguyên nhân chưa được xác định.

Hai ca nên ưu tiên kiểm tra ở vòng sau là xe rất nhỏ ở xa và xe bị cắt ở mép ảnh hoặc bị che khuất. Các trường hợp này có chi phí rà nhãn cao vì khó xác định chính xác phạm vi bounding box. Đồng thời cần tránh chọn hai ảnh quá sát nhau về thời gian vì chúng gần như mô tả cùng một cảnh và không mang lại nhiều thông tin mới.

Tập kiểm thử chỉ có 20 ảnh và 403 box tham chiếu, trong đó 14 box cao dưới 16 px bị bỏ qua. Vì vậy kết quả đánh giá có thể chưa đại diện đầy đủ cho toàn bộ dữ liệu. Ngoài ra, luật đánh giá bỏ qua xe quá nhỏ nên kết quả không phản ánh đầy đủ khả năng phát hiện các xe ở rất xa.

Một giới hạn quan trọng khác là nhãn tham chiếu ban đầu được tạo bởi model và chưa được người kiểm tra thủ công toàn bộ. Vì vậy sự khác biệt giữa prediction và reference có thể đến từ lỗi của model, lỗi của nhãn tham chiếu hoặc cả hai. Do đó không nên coi nhãn test do model tạo là chân lý tuyệt đối.

Nếu AP50 giảm như ở vòng 1, trước khi train thêm cần kiểm tra lại các bounding box đã sửa trong `REVIEW_LOG.csv`, đối chiếu với `round1_diff.md`, xem lại các frame được chọn có quá gần nhau hay không và kiểm tra những trường hợp mà nhãn tham chiếu có thể sai. Chỉ sau khi xác định nguyên nhân mới nên quyết định có tiếp tục thêm một vòng active learning hay không.