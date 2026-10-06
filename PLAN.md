# Kế hoạch project: Đếm đối tượng đi qua một đoạn phố từ luồng EarthCam

Môn: Trí tuệ nhân tạo và ứng dụng. Máy train: RTX 3060 Laptop 6GB.

## 0. Tóm tắt một dòng

Fine-tune **YOLO** để phát hiện người trên đúng camera đã chọn, dùng **ByteTrack** gán ID xuyên frame, rồi đếm mỗi ID **đúng một lần** khi quỹ đạo của nó **cắt vạch ảo** (hoặc đi xuyên vùng đa giác) đặt tại vạch qua đường / miệng ngõ. Đánh giá bằng sai số đếm so với đếm tay trên các clip quay ở thời điểm khác thời điểm train.

## 1. Phân tích yêu cầu (bài giảng 7.1, bước 1)

### 1.1 Xác định vấn đề
- Bài toán là **đếm lưu lượng (flow counting)**: số đối tượng đi qua một ranh giới trong một khoảng thời gian. Đây không phải đếm số đối tượng có mặt trong một frame.
- Hệ quả: chỉ detection là không đủ. Một người xuất hiện trong 300 frame vẫn chỉ được tính 1. Bắt buộc cần **tracking** để giữ danh tính, và cần **luật đếm** gắn với ranh giới.

### 1.2 Đầu vào và đầu ra
| | Nội dung |
|---|---|
| Đầu vào | Luồng video EarthCam (trực tiếp hoặc file ghi lại), toạ độ vạch đếm A-B, khoảng thời gian đếm |
| Đầu ra | Số người đi qua theo từng chiều (A sang B, B sang A) và tổng, video có vẽ box, ID, vạch, bộ đếm |

### 1.3 Ràng buộc và điều kiện
- **Thầy chọn thời điểm bất kỳ**: mô hình phải chịu được sáng, trưa, chiều tối, đêm, mưa. Đây là rủi ro lớn nhất, quyết định cách thu dữ liệu.
- **Thời gian thực**: luồng khoảng 25-30 fps. Nếu xử lý chậm hơn luồng mà bỏ frame lung tung thì tracker đứt ID và đếm sai.
- **Tài nguyên**: GPU 6GB, nên dùng model cỡ n hoặc s.
- **Fact check bằng đếm tay**: luật đếm phải đơn giản, người đếm tay áp dụng được y hệt.

### 1.4 Yêu cầu đã chốt (06/10/2026)
| Hạng mục | Quyết định |
|---|---|
| Lớp cần đếm | **Chỉ người** (một lớp `person`) |
| Chiều đi | **Có phân biệt chiều**: báo riêng A→B và B→A |
| Đi qua rồi quay lại | **Chỉ tính 1**, theo chiều của lần cắt vạch đầu tiên |
| Ngưỡng chấp nhận | Sai số đếm **không quá 5%**, tính riêng cho từng chiều |

Hệ quả của các quyết định này:
- **Luật "chỉ tính 1" chỉ áp dụng khi người đó còn trong khung hình.** Người đã ra khỏi khung rồi quay lại sẽ thành ID mới và được tính lần nữa. Người đếm tay cũng không thể nhận ra họ, nên hai bên vẫn nhất quán. Cần nói rõ điều này với thầy.
- **Sai số 5% cần mẫu đủ lớn.** Clip có 20 lượt thì sai 1 lượt đã là 5%. Nên đề nghị thầy đếm trong khoảng có **ít nhất 40-60 lượt mỗi chiều**, hoặc kéo dài thời gian đếm cho đủ.
- Ngưỡng 5% chặt hơn dự kiến nên ưu tiên model cỡ s, ảnh vào 960 trở lên, và dùng hai vạch song song để chống đếm sai khi người đứng lại trên vạch.
- Chỉ một lớp nên gán nhãn nhanh hơn nhiều; phương tiện không cần gán nhãn.

## 2. Camera (đã chốt 06/10/2026)

**Camera chính: Songridan-gil, Seoul.** Dự phòng: Kathy Pub, Gangnam, Seoul. Cấu hình ở `configs/cameras.json`, ảnh mẫu ở `data/snapshots/`.

- Luồng 720p, camera cố định, góc cao, ngã ba có 3 vạch qua đường.
- Seoul đi trước Việt Nam 2 giờ, không đổi giờ mùa hè.
- Thầy quay video ở thời điểm bất kỳ, nên dữ liệu train phải phủ: sáng, trưa, chiều tà, tối có đèn, đêm khuya, trời mưa, ngày thường và cuối tuần.

### 2.1 Vạch đếm (đã chốt 06/10/2026)
**Vạch qua đường phía trên**, nối góc quán cà phê bên trái khung hình với vỉa hè nhà hàng phía xa. Ảnh minh hoạ: `data/snapshots/songridan_vach_dem.jpg`. Cấu hình: `configs/line_songridan.json`.

- Vạch ảo A-B: A = (565, 329), B = (770, 413) trên khung 1280x720. Vạch ảo nằm giữa lòng đường, song song với các sọc trắng, tức là vuông góc với hướng người đi.
- Toạ độ được lưu thêm dạng chuẩn hoá theo vùng video, để dùng được với video quay màn hình ở cỡ bất kỳ.
- Hai chiều: **"tới nhà hàng"** (từ góc quán cà phê sang vỉa hè nhà hàng) và **"về quán cà phê"** (chiều ngược lại).
- **Chỉ tính người đi bộ.** Người ngồi trên xe máy, xe đạp, xe lăn điện không tính.

Lý do chọn vạch này, dựa trên 4 phút quan sát luồng lúc 17:45 giờ Seoul:
- Nhiều người qua nhất trong các vạch, cần thiết để đủ mẫu cho ngưỡng 5%.
- Cả hai đầu vạch và hai vỉa hè đều nằm trọn trong khung, nên mỗi người được bám đủ lâu trước và sau khi cắt vạch.
- Không có vật che cố định. Hai vạch phía dưới bị cần camera giao thông chắn và bị cắt ở mép khung hình.
- Điểm yếu: người tại vạch chỉ cao khoảng 75-80 pixel, và ô tô từ nhánh đường trên bên trái chạy cắt qua vạch. Nếu baseline tuần 2 cho thấy sai số cao thì phương án thay thế là vạch lớn bên trái phía dưới, nơi người cao khoảng 120 pixel.

### 2.2 Điểm khó riêng của camera này
- Ô tô, xe tải dừng hoặc chạy qua che người đang sang đường.
- Nhóm người đứng chờ trước cửa nhà hàng, sát đầu vạch. Vạch ảo đặt giữa lòng đường nên họ không bị tính.
- Xe máy giao hàng và xe đạp chạy cắt vạch: phải loại bằng cách không gán nhãn người ngồi trên xe, hoặc lọc theo lớp xe trùng vị trí.
- Luồng 720p, khi thầy quay màn hình có thể còn nhỏ hơn. Cắt vùng quanh vạch rồi phóng to trước khi đưa vào model.

### 2.3 Công cụ quay: `src/capture.py`
```
python3 src/capture.py list                                   # camera, giờ tại camera, các phiên đã quay
python3 src/capture.py snapshot --cam songridan               # chụp 1 khung hình
python3 src/capture.py screen --cam songridan --minutes 15    # quay màn hình (để trình phát toàn màn hình)
python3 src/capture.py screen --cam songridan --pick          # click chọn cửa sổ cần quay
python3 src/capture.py stream --cam songridan --minutes 15    # đọc thẳng luồng, không chiếm màn hình
```
- Mỗi phiên sinh file `.mkv` và file `.json` ghi giờ quay, giờ tại camera, buổi (sáng/trưa/chiều/đêm), vùng quay, fps. Tên file có sẵn buổi để chia tập theo phiên.
- Chế độ `screen` quay 25 fps, không quay con trỏ chuột. Luôn để trình phát **toàn màn hình 1920x1080** để mọi phiên giống nhau.
- Chế độ `stream` giữ nguyên chất lượng gốc, khoảng 100 MB cho 15 phút, tự nối lại khi luồng đứt. Dùng khi không tiện để máy quay màn hình.
- **Đã chốt (06/10/2026): đầu vào lúc bảo vệ là video thầy tự quay màn hình.** Vì vậy dữ liệu train chủ yếu phải là video quay màn hình (`screen`). Chế độ `stream` chỉ dùng để bổ sung dữ liệu khi không tiện để máy quay màn hình.
- Không biết trước thầy quay thế nào (độ phân giải, cửa sổ to hay nhỏ, có dính giao diện trình duyệt không). Cách đối phó:
  - Quay dữ liệu train ở **nhiều cỡ cửa sổ**: toàn màn hình, nửa màn hình, có và không có thanh trình duyệt.
  - Bật tăng cường co giãn (scale) và làm mờ, nén khi train.
  - Chương trình có bước **chọn vùng video và vẽ vạch đếm trên chính video của thầy** trước khi chạy, không dùng toạ độ cố định.
  - Nên xin thầy một video mẫu, hoặc hỏi thầy quay bằng gì, để quay dữ liệu train giống hệt.

### 2.4 Khảo sát thêm 7 luồng (06/10/2026)
| Camera | Luồng | Nhận xét từ khung hình thật | Xếp hạng |
|---|---|---|---|
| **Songridan-gil, Seoul** | 720p | Góc cao, ngã ba có 3 vạch qua đường nằm trọn trong khung, người đi đều, gần như không bị che | **1, camera chính** |
| **Kathy Pub, Gangnam, Seoul** | 720p, 15 fps | Hẻm đi bộ, người to và rõ, lượt qua lại đều. Biển hiệu và cột điện che một phần, khuôn mặt bị làm mờ sẵn | **2, dự phòng** |
| Abbey Road, London | 1080p, 30 fps | Vạch qua đường ở xa, người nhỏ, khách du lịch đứng tụm và tạo dáng trên vạch. Xe tải đỗ che gần hết vỉa hè cận cảnh trong lần xem thứ hai | 3 |
| Myrtle Beach, North View | 1080p | Lối đi lát gỗ rộng, góc cao, đèn đêm tốt, không có xe. Chỉ xem được lúc 4 giờ sáng, không có người; mùa thu là mùa vắng | 4, chưa kiểm chứng |
| Gangnam Cam, Seoul | 720p, 15 fps | Mái hiên che một phần ba khung hình, người ở xa, không có vạch rõ | Loại |
| Madison, Indiana | 720p | Ống kính mắt cá, cây che giữa khung, thị trấn nhỏ rất thưa người | Loại |
| Florence, South Carolina | 720p | Cần đèn giao thông chắn giữa khung, tối, thưa người | Loại |
| Seaside Heights, New Jersey | không rõ | Luồng từ chối kết nối (lỗi 403) ở cả hai lần thử | Không đánh giá được |

Ảnh so sánh 2 phút: `data/snapshots/songridan_2phut.jpg`, `kathypub_2phut.jpg`, `abbeyroad_2phut.jpg`.

Lý do Songridan-gil hơn Abbey Road:
- Vạch qua đường nằm gần và giữa khung, người tại vạch to hơn dù luồng chỉ 720p.
- Ít vật che cố định, không có xe đỗ chắn ống kính.
- Người đi qua liên tục, không đứng lại chụp ảnh trên vạch. Đây là kiểu lỗi khó xử lý nhất với ngưỡng 5%.
- Seoul chỉ lệch Việt Nam 2 giờ, nên quay dữ liệu ban ngày rất tiện và video của thầy nhiều khả năng cũng là ban ngày.
- Điểm yếu: 720p, ô tô và taxi thỉnh thoảng che vạch, phần cần camera giao thông chắn góc dưới khung.

## 3. Kiến trúc thuật toán

```
Luồng EarthCam ──> Lấy frame ──> Cắt ROI ──> YOLO (detect) ──> ByteTrack (gán ID)
                                                                    │
                    Video kết quả + số đếm <── Luật đếm cắt vạch <──┘
```

### 3.1 Lấy frame
- **Cách chính**: quay màn hình trình duyệt (OBS hoặc ffmpeg x11grab) ở độ phân giải cố định, cùng một vùng cửa sổ, cùng fps.
- **Nguyên tắc vàng**: đường lấy ảnh lúc train và lúc bảo vệ phải **giống hệt nhau** (cùng độ phân giải, cùng tỉ lệ, cùng cách nén). Lệch đường lấy ảnh là lệch miền dữ liệu.
- Cách thay thế: lấy địa chỉ HLS (.m3u8) trong tab Network của trình duyệt rồi đọc bằng OpenCV/ffmpeg. Chất lượng tốt hơn nhưng địa chỉ hay hết hạn, chỉ nên là phương án phụ.
- Khi chạy trực tiếp: một luồng (thread) đọc frame, một luồng xử lý, đánh dấu thời gian từng frame để biết có rớt frame không.

### 3.2 Phát hiện: YOLO fine-tune (bài giảng 6.1)
- Chọn **one-stage detector** vì cần thời gian thực. Bài giảng đã so sánh two-stage (Faster R-CNN) chính xác nhưng chậm với one-stage (YOLO, SSD) nhanh.
- Model: **YOLO26s** là lựa chọn chính vì ngưỡng 5%, **YOLO26n** (trùng ví dụ trong bài giảng 7.2) để so sánh tốc độ, qua thư viện Ultralytics.
- **Transfer learning**: khởi tạo từ trọng số COCO, fine-tune trên dữ liệu của camera đã chọn.
- Kích thước ảnh vào 960 hoặc 1280 vì đối tượng nhỏ; chỉ đưa **vùng ROI quanh vạch đếm** vào model để đối tượng to hơn và chạy nhanh hơn.
- Lý thuyết cần trình bày: lưới dự đoán, anchor/anchor-free, FPN + PAN, hàm loss (box CIoU, class, objectness), NMS, IoU, mAP.

### 3.3 Bám đối tượng: ByteTrack (bài giảng 6.3)
- **Tracking-by-detection**: Kalman filter dự đoán vị trí frame kế tiếp, ghép detection với track bằng IoU và thuật toán Hungarian.
- Điểm riêng của ByteTrack: ghép lần hai với các **box điểm thấp**, nên người bị che một phần không mất ID. Đây đúng là tình huống ở vạch qua đường.
- Không cần mạng ReID nên nhẹ, hợp GPU 6GB.
- Thí nghiệm so sánh (ablation) để có nội dung báo cáo: Centroid tracker, IoU tracker, SORT, ByteTrack, BoT-SORT trên cùng bộ clip test.

### 3.4 Luật đếm cắt vạch (phần thuật toán tự viết, là lõi của project)

Vạch ảo là đoạn thẳng A-B. Với mỗi track, lấy **điểm neo P = trung điểm cạnh đáy của box** (vị trí chân, ổn định hơn tâm box).

Phía của P so với vạch tính bằng dấu tích có hướng:

```
side(P) = sign( (B.x - A.x)*(P.y - A.y) - (B.y - A.y)*(P.x - A.x) )
```

Một track được đếm khi thoả **đồng thời**:
1. `side` đổi dấu giữa hai thời điểm, **và** đoạn P_trước → P_sau thật sự cắt đoạn A-B (không phải cắt phần kéo dài của vạch).
2. Track đã tồn tại ít nhất N frame (ví dụ 5), loại box nhiễu chớp nháy.
3. Sau khi sang phía mới, P cách vạch hơn δ pixel (**vùng trễ / hysteresis**), loại người đứng rung ngay trên vạch.
4. ID chưa nằm trong tập đã đếm.

Chiều đi suy ra từ dấu thay đổi (âm sang dương là A→B, ngược lại là B→A).

```
counted = set(); last_side = {}
for mỗi frame:
    tracks = bytetrack.update(yolo(frame[ROI]))
    for t in tracks:
        P = chân_box(t); d = khoảng_cách_có_dấu(P, AB)
        if |d| < δ: continue                    # trong vùng trễ, chưa kết luận
        s = sign(d)
        if t.id in last_side and s != last_side[t.id] \
           and t.age >= N and cắt_trong_đoạn(t, AB) and t.id not in counted:
            count[chiều(s)] += 1; counted.add(t.id)
        last_side[t.id] = s
```

Biến thể cho **miệng ngõ**: dùng vùng đa giác thay vạch, đếm khi track vào từ một cạnh và ra ở cạnh đối diện. Biến thể bền hơn: hai vạch song song, phải cắt cả hai mới tính.

Xử lý lỗi đã biết:
- **Đổi ID ngay tại vạch** làm đếm đôi hoặc hụt: tăng `track_buffer`, và ghép lại track mới sinh sát vị trí track vừa mất.
- **Nhóm người dính nhau**: đặt vạch ở chỗ thoáng, góc nhìn rõ nhất.
- **Rớt frame**: luôn xử lý với bước frame cố định (ví dụ mọi frame thứ 2), không bỏ ngẫu nhiên.

## 4. Dữ liệu (bài giảng 7.1, bước 2)

### 4.1 Thu thập
- Quay nhiều phiên **trải đều điều kiện**: sáng, trưa, chiều tà, đêm, trời mưa hoặc âm u, ngày thường và cuối tuần. Mỗi phiên 10-20 phút. Mục tiêu khoảng 12-16 phiên.
- Đây là cách duy nhất chống lại việc thầy chọn giờ bất kỳ.

### 4.2 Chia tập (tránh rò rỉ dữ liệu)
- Chia **theo phiên quay**, không trộn frame ngẫu nhiên. Hai frame cạnh nhau gần như giống hệt, trộn vào sẽ cho mAP ảo.
- Train khoảng 70%, validation 15%, test 15% số phiên. Tập test phải có đủ ngày và đêm.
- Giữ riêng **bộ clip đếm**: 10-15 clip, mỗi clip 3-5 phút, quay ở ngày khác hẳn, chỉ dùng để chấm sai số đếm.

### 4.3 Gán nhãn
- Trích 1 frame mỗi 1-2 giây, mục tiêu **1500-3000 ảnh**.
- **Gán nhãn bán tự động**: cho YOLO pretrained cỡ lớn (x) dự đoán trước, rồi sửa tay trong CVAT hoặc Roboflow. Nhanh hơn 5-10 lần so với vẽ từ đầu.
- Lớp: chỉ `person`, và chỉ người đi bộ. Người ngồi trên xe máy, xe đạp, xe lăn điện không gán nhãn.
- Quy ước: vẽ box cả đối tượng bị che một phần; bỏ qua đối tượng quá xa ngoài ROI.
- Bộ clip đếm: **hai người đếm tay độc lập**, lệch nhau thì xem lại. Ghi số theo từng chiều.

### 4.4 Tăng cường dữ liệu
Mosaic, lật ngang, đổi HSV (giả lập thay đổi ánh sáng), scale, làm mờ nhẹ và nhiễu nén (giả lập chất lượng stream). Tắt mosaic 10 epoch cuối.

## 5. Huấn luyện (bài giảng 7.1, bước 3 và bài giảng 5)

| Tham số | Giá trị khởi điểm |
|---|---|
| Model | yolo26n, yolo26s (pretrained COCO) |
| imgsz | 960 (thử 1280 nếu đủ VRAM) |
| Batch | 8-16 (theo VRAM 6GB), bật AMP |
| Epoch | 100, early stopping patience 20 |
| Optimizer | SGD momentum 0.937 hoặc AdamW, lr0 0.01, cosine, warmup 3 epoch |
| Weight decay | 0.0005 |

- Theo dõi đường loss train/val và mAP để phát hiện **overfitting**; điều kiện dừng theo bài giảng: loss hội tụ, mAP validation không tăng.
- Thí nghiệm cần chạy: pretrained không fine-tune (baseline) so với fine-tune; cỡ n so với s; imgsz 640 so với 960.
- **Vòng lặp cải thiện**: chạy model trên clip mới, gom các frame đếm sai (đêm, che khuất), gán nhãn bổ sung, train lại. Lặp 2-3 vòng.

## 6. Tối ưu mô hình (bài giảng 7.2)

- Xuất ONNX rồi TensorRT **FP16**: thường nhanh gấp đôi, gần như không mất độ chính xác.
- Thử **INT8 post-training quantization** với tập calibration lấy từ dữ liệu train.
- Tuỳ thời gian: pruning + knowledge distillation (teacher cỡ s, student cỡ n) như ví dụ yolo26n trong bài giảng.
- Bảng báo cáo: Params, GFLOPs, FPS, độ trễ, mAP, sai số đếm cho từng phiên bản.

## 7. Đánh giá (bài giảng 7.1, bước 5)

| Tầng | Chỉ số | Dữ liệu |
|---|---|---|
| Detector | Precision, Recall, mAP@0.5, mAP@0.5:0.95 | Tập test đã gán nhãn |
| Tracker | Số lần đổi ID, MOTA, IDF1 (trên 1-2 clip ngắn gán nhãn ID, tuỳ chọn) | Clip ngắn |
| **Đếm (chỉ số chính)** | Sai số tương đối = \|đếm máy − đếm tay\| / đếm tay, tính riêng từng chiều, mục tiêu ≤ 5% | Bộ clip đếm |
| Hệ thống | FPS đầu-cuối, tỉ lệ rớt frame | Chạy trực tiếp |

- Báo cáo sai số đếm tách theo điều kiện: ngày, đêm, đông, vắng.
- Lưu ý: sai số tổng có thể đẹp nhờ **đếm thừa bù đếm hụt**. Phải xem lại video kết quả và thống kê riêng số đếm thừa, số đếm hụt.
- Phân tích lỗi: mỗi ca sai thuộc nhóm nào (hụt detection, đổi ID, che khuất, đứng trên vạch) và đã xử lý ra sao.

## 8. Triển khai và kịch bản bảo vệ (bài giảng 7.1, bước 6)

- Ứng dụng: script Python + OpenCV, hoặc giao diện Streamlit/Gradio. Chức năng: chọn nguồn (màn hình / file / URL), **vẽ vạch bằng chuột**, nút bắt đầu/dừng, hiển thị bộ đếm trực tiếp, lưu video kết quả và file log CSV (thời điểm, ID, lớp, chiều).
- **Kịch bản bảo vệ đề xuất**: khi thầy bấm giờ, chương trình vừa đếm trực tiếp vừa **ghi lại đúng đoạn video đó**. Thầy đếm tay trên cùng đoạn. Nếu có tranh cãi, tua lại video kết quả để đối chiếu từng lần đếm.
- Chuẩn bị trước: vạch đã cấu hình sẵn cho camera chính và dự phòng, kiểm tra mạng, có sẵn video demo ghi trước phòng khi luồng sập.

## 9. Lộ trình (khoảng 7 tuần, co giãn theo hạn nộp)

| Tuần | Việc | Kết quả |
|---|---|---|
| 1 | **Xong 06/10**: chốt yêu cầu, chốt camera Songridan-gil, dựng công cụ quay. Còn lại: quay 2-3 phiên thử, xin thầy video mẫu | Mục 1.4, mục 2, `src/capture.py` |
| 2 | Baseline: YOLO pretrained + ByteTrack + luật cắt vạch, chạy trên clip thử | Pipeline chạy được đầu-cuối, sai số baseline |
| 3 | Thu đủ dữ liệu các điều kiện, đếm tay bộ clip đếm | 12-16 phiên, 10-15 clip có số đếm tay |
| 4 | Gán nhãn bán tự động, chia tập theo phiên | Bộ dữ liệu 1500-3000 ảnh |
| 5 | Fine-tune, so sánh model và tracker, tinh chỉnh luật đếm | Bảng thí nghiệm, model tốt nhất |
| 6 | Tối ưu (TensorRT FP16/INT8), dựng ứng dụng, chạy thử trực tiếp nhiều khung giờ | Ứng dụng demo, bảng tốc độ |
| 7 | Phân tích lỗi, vòng bổ sung dữ liệu, viết báo cáo, slide, tập bảo vệ | Báo cáo, slide, video demo |

Tuần 2 làm baseline trước khi gán nhãn là có chủ đích: biết sớm camera có khả thi không, và biết lỗi nằm ở đâu để gán nhãn đúng chỗ.

## 10. Cấu trúc thư mục

```
Project Earthcam/
├── PLAN.md
├── data/
│   ├── raw/            # video quay theo phiên: 2026-10-10_morning.mp4 ...
│   ├── frames/         # frame trích ra
│   ├── dataset/        # images + labels định dạng YOLO, data.yaml
│   └── count_clips/    # clip đếm + ground_truth.csv
├── src/
│   ├── capture.py      # quay màn hình / đọc stream
│   ├── extract_frames.py
│   ├── autolabel.py    # gán nhãn sơ bộ bằng model lớn
│   ├── train.py
│   ├── counter.py      # luật đếm cắt vạch
│   ├── run.py          # pipeline detect + track + count
│   ├── evaluate_count.py
│   └── app.py          # giao diện demo
├── configs/            # line.json theo camera, bytetrack.yaml
├── runs/               # kết quả train
└── report/
```

Công cụ: Python 3.10, PyTorch, Ultralytics, OpenCV, supervision (tuỳ chọn), ffmpeg/OBS, CVAT hoặc Roboflow, TensorRT.

## 11. Bảng đối chiếu kiến thức môn học

| Bài giảng | Dùng ở đâu trong project |
|---|---|
| Phần 3, Học có giám sát | Bài toán phân loại + hồi quy box, chia train/val/test, overfitting |
| Phần 4, Học không giám sát | K-means chọn anchor (YOLOv2-v5), phân cụm để phân tích dữ liệu |
| Phần 5, Học sâu | CNN, backprop, optimizer, batch norm, hàm kích hoạt |
| Phần 6.1, Object detection | YOLO, NMS, IoU, mAP, FLOPs, thời gian suy luận |
| Phần 6.3, Tracking | Kalman, Hungarian, SORT, DeepSORT, ByteTrack, MOTA, IDF1 |
| Phần 7.1, Quy trình triển khai | Khung 6 bước của toàn bộ kế hoạch này |
| Phần 7.2, Nén mô hình | Quantization, pruning, distillation, bảng tốc độ |
| Phần 2, Tìm kiếm | Ghép cặp tối ưu (Hungarian) là bài toán gán, có thể liên hệ |

## 12. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Thầy chọn giờ đêm hoặc mưa | Dữ liệu train phủ đủ điều kiện, báo cáo sai số theo từng điều kiện |
| Camera sập hoặc đổi góc | Dự phòng Kathy Pub, công cụ vẽ lại vạch trong 10 giây |
| Máy xử lý không kịp luồng | ROI nhỏ, model n, TensorRT FP16, bước frame cố định |
| Che khuất khi đông | Camera góc cao, vạch đặt chỗ thoáng, ByteTrack, hai vạch song song |
| Đếm tay và máy hiểu luật khác nhau | Chốt định nghĩa "đi qua" bằng văn bản từ tuần 1, vẽ vạch lên video cho thầy thấy |
| Chất lượng quay màn hình khác lúc train | Cố định độ phân giải, vùng quay, trình duyệt; train bằng chính đường quay đó |

## 13. Dàn ý báo cáo

1. Đặt vấn đề và phân tích yêu cầu
2. Cơ sở lý thuyết: YOLO, ByteTrack (Kalman, Hungarian), luật đếm hình học
3. Dữ liệu: camera, thu thập, gán nhãn, chia tập
4. Huấn luyện và thí nghiệm so sánh
5. Tối ưu mô hình
6. Kết quả: mAP, sai số đếm, tốc độ, phân tích lỗi
7. Triển khai, demo, hạn chế và hướng phát triển
