# Earthcam Detection

Project môn Trí tuệ nhân tạo và ứng dụng: đếm số người đi bộ đi qua một vạch qua đường trên luồng EarthCam Songridan-gil (Seoul), có phân biệt chiều.

Kế hoạch đầy đủ: [PLAN.md](PLAN.md). Hiện mới có phần thu dữ liệu.

## Cài đặt

Cần Python 3.9 trở lên và ffmpeg.

**Windows**
```
winget install Gyan.FFmpeg
pip install -r requirements.txt
```
Mở lại terminal sau khi cài ffmpeg để nhận PATH.

**Ubuntu (X11)**
```
sudo apt install ffmpeg x11-utils
```

## Quay dữ liệu

Trên Windows gõ `python`, trên Linux gõ `python3`.

```
python src/capture.py list
python src/capture.py screen --cam songridan --minutes 15 --delay 10
python src/capture.py screen --cam songridan --minutes 15 --delay 10 --pick
python src/capture.py stream --cam songridan --minutes 15
python src/capture.py snapshot --cam songridan
```

- `screen`: quay màn hình. Trong lúc đếm ngược, chuyển sang trình duyệt và bật trình phát toàn màn hình.
- `screen --pick`: chỉ quay một cửa sổ. Linux: click vào cửa sổ. Windows: cửa sổ đang ở trên cùng khi hết đếm ngược.
- `stream`: đọc thẳng luồng, không chiếm màn hình.
- Dừng sớm bằng Ctrl+C. Thêm `--tag mua` hoặc `--tag cuoituan` để ghi chú.

Video và file thông tin phiên nằm ở `data/raw/<camera>/`. Thư mục `data/` không đưa lên git.

## Cấu hình

- `configs/cameras.json`: danh sách camera.
- `configs/line_songridan.json`: vạch đếm, tên hai chiều, luật đếm.

Video chỉ dùng cho mục đích học tập, không phát tán lại.
