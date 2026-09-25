# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 24

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 
1. Vị trí: Góc dưới bên phải
   - Mô tả: Xe bị cắt một phần bởi mép ảnh, thân xe nằm trong vùng tối và bị nhòe.
   - Lỗi AI có thể gặp: Dễ bỏ sót xe hoặc vẽ bounding box không bao phủ đúng phần xe nhìn thấy.

2. Vị trí: Khu vực giữa ảnh, hơi bên trái
   - Mô tả: Xe đang đi về phía camera, đèn pha rất sáng trong khi thân xe khá tối.
   - Lỗi AI có thể gặp: Dễ xác định sai kích thước xe hoặc chỉ tập trung vào vùng đèn pha khi vẽ bounding box.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
