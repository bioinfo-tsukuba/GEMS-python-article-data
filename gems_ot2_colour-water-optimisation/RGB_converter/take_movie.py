import os
import cv2
import time
import argparse
from datetime import datetime

def capture_with_retry(cap, retries=3, wait=5):
    """
    cap.read() の結果が False のときに、最大 retries 回まで
    wait 秒待って再試行する。成功したら (True, frame)、失敗続きなら (False, None) を返す。
    """
    for attempt in range(1, retries + 1):
        ret, frame = cap.read()
        if ret:
            return True, frame
        else:
            if attempt < retries:
                print(f"フレーム取得失敗 (試行 {attempt}/{retries})。{wait}秒後に再試行します...")
                time.sleep(wait)
    # all retries failed
    return False, None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Webカメラでタイムラプスを撮影し、右下にタイムスタンプを合成表示するスクリプト"
    )
    parser.add_argument(
        "--duration", type=float, required=True,
        help="撮影する合計時間（秒単位）。必須項目。"
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="フレームを取得する間隔（秒）。デフォルトは10秒。"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="撮影するフレーム数。指定するとこちらを優先し、間隔は自動計算される。"
    )
    parser.add_argument(
        "--output", type=str, default="timelapse_with_timestamp.mp4",
        help="出力する動画ファイル名。デフォルトは 'timelapse_with_timestamp.mp4'。"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="使用するカメラのデバイス番号。デフォルトは0。"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    total_duration = args.duration
    interval = args.interval
    num_frames = args.frames
    output_filename = args.output
    camera_index = args.camera

    # フレーム数指定時は間隔を再計算
    if num_frames is not None:
        if num_frames <= 0:
            print("エラー：--frames は 1以上の整数を指定してください。")
            return
        interval = total_duration / num_frames
        print(f"フレーム数 {num_frames} を優先。間隔を {interval:.3f} 秒に設定します。")
    else:
        if interval <= 0:
            print("エラー：--interval は 0より大きい値を指定してください。")
            return
        estimated_frames = int(total_duration // interval)
        print(f"フレーム数未指定。間隔 {interval:.3f} 秒で約 {estimated_frames} フレーム撮影します。")

    # カメラを開く
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"カメラ（デバイス {camera_index}）を開けませんでした。")
        return

    # 最初のフレームを取得してサイズを決定
    ok, frame = capture_with_retry(cap, retries=3, wait=5)
    if not ok:
        print("最初のフレーム取得に 3 回失敗しました。終了します。")
        cap.release()
        return
    height, width = frame.shape[:2]

    # → 変更ポイント：コーデックを 'mp4v' に、出力コンテナを .mp4 に対応
    #    もし H.264（環境によっては 'H264' や 'X264'）が使えるならそちらでも構いません。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 出力FPSは再生時の滑らかさを重視して 30fps に固定
    output_fps = 30
    writer = cv2.VideoWriter(output_filename, fourcc, output_fps, (width, height))
    # ---------------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    margin = 10

    print("タイムラプス撮影を開始します。終了はCtrl+Cまたは撮影時間終了まで。")
    start_time = time.time()
    frames_captured = 0

    try:
        if num_frames is not None:
            # フレーム数優先モード
            for i in range(num_frames):
                ok, frame = capture_with_retry(cap, retries=3, wait=5)
                if not ok:
                    print("フレーム取得に 3 回失敗しました。ループを抜けます。")
                    break
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                (text_width, text_height), baseline = cv2.getTextSize(timestamp, font, font_scale, thickness)
                x = width - text_width - margin
                y = height - baseline - margin

                cv2.putText(frame, timestamp, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, timestamp, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                writer.write(frame)
                frames_captured += 1

                if i < num_frames - 1:
                    time.sleep(interval)

        else:
            # 間隔優先モード
            next_capture_time = start_time
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed >= total_duration:
                    break

                if current_time < next_capture_time:
                    time.sleep(next_capture_time - current_time)

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                ok, frame = capture_with_retry(cap, retries=3, wait=5)
                if not ok:
                    print("フレーム取得に 3 回失敗しました。ループを抜けます。")
                    break

                (text_width, text_height), baseline = cv2.getTextSize(timestamp, font, font_scale, thickness)
                x = width - text_width - margin
                y = height - baseline - margin

                cv2.putText(frame, timestamp, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, timestamp, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                writer.write(frame)
                frames_captured += 1

                next_capture_time += interval

    except KeyboardInterrupt:
        print("\nユーザーによる中断。撮影を終了します。")
    finally:
        cap.release()
        writer.release()

    end_time = time.time()
    actual_duration = end_time - start_time
    print(f"撮影終了：実際に撮影したフレーム数 = {frames_captured}、撮影時間 = {actual_duration:.2f} 秒")
    print(f"出力ファイル：{output_filename}")

if __name__ == "__main__":
    main()
