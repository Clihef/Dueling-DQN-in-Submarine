### 参数配置
- 运行 `py main.py --help` 查看
   **usage**: 
   main.py [-h] [--dynamic-plot] [--live-plot][--animation-interval-ms][--save-trajectory-data]


    **options**:
    -h, --help              show this help message and exit
    --dynamic-plot          生成并保存动态轨迹 GIF（默认关闭）
    --live-plot             窗口实时播放动态轨迹（不保存 GIF）
    --animation-interval-ms 动态轨迹图帧间隔（毫秒），默认 60ms
    --save-trajectory-data  保存 trajectory_data.json（默认关闭）