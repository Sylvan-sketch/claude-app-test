# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def crop_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片没找到，检查一下路径！当前路径是: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("没检测到人脸，换张正脸清晰的图试试。")
        return

    (x, y, w, h) = faces[0]

    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)

    face_crop = img[y:y + h, x:x + w]
）
    img_rgb = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Cropped Face")
    plt.imshow(face_rgb)
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    crop_face('test_face.jpg')