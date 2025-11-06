import time
import cv2

DEVICE_ID = 0

def camera():
    #カメラの設定　デバイスIDは0
    cap = cv2.VideoCapture(DEVICE_ID)

    # ひたすら画像を取得し続ける
    #繰り返しのためのwhile文
    start = time.time()
    while True:
        #カメラからの画像取得
        ret, frame = cap.read()

        #カメラの画像の出力
        cv2.imshow('camera' , frame)

        #終了のためのキー入力待ち
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #メモリを解放して終了するためのコマンド
    cap.release()
    cv2.destroyAllWindows()

def take_picture(retry_num: int = 5, focus_time: float = 5.0, save_path: str = None):
    ret_val = None
    save_path = str(save_path) if save_path else None
    cap = cv2.VideoCapture(DEVICE_ID)
    start = time.time()
    while True:
        ret, frame = cap.read()
        if time.time() - start > focus_time:
            for _ in range(retry_num):
                ret, frame = cap.read()
                if ret:
                    if save_path:
                        cv2.imwrite(save_path, frame)
                        print(f"Saved the image to {save_path}")
                        ret_val = save_path
                    else:
                        ret_val = frame
                    break
            break
    cap.release()
    cv2.destroyAllWindows()
    return ret_val

if __name__ == '__main__':
    # camera()
    pic = take_picture(focus_time=5.0, save_path='ot2_experiment/step_00000007/well_plate.jpg')
    # take_picture_with_focus()