import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

emotion_labels = {
    0: '生气 (Angry)',
    1: '厌恶 (Disgust)',
    2: '恐惧 (Fear)',
    3: '开心 (Happy)',
    4: '难过 (Sad)',
    5: '惊讶 (Surprise)',
    6: '平静 (Neutral)'
}


def count_emotions(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件：{csv_path}")
        print("请检查：1. 文件是否放在了 data 文件夹下？ 2. 文件名是否正确？")
        return

    print("正在读取数据，请稍候...")
    df = pd.read_csv(csv_path)

    emotion_counts = df['emotion'].value_counts()

    total_images = len(df)

    print("\n" + "=" * 30)
    print("📊 FER2013 数据集表情分布统计")
    print("=" * 30)
    print(f"总图片数量: {total_images} 张\n")


    for emotion_id in range(7):
        count = emotion_counts.get(emotion_id, 0)

        percentage = (count / total_images) * 100

        label_name = emotion_labels[emotion_id]

        print(f"{label_name:<15}: {count:>5} 张 (占比 {percentage:.2f}%)")

    print("=" * 30)


if __name__ == '__main__':
    csv_file_path = 'data/fer2013.csv'
    count_emotions(csv_file_path)


csv_file_path = 'data/fer2013.csv'

if not os.path.exists(csv_file_path):
    print(f"❌ 找不到文件：{csv_file_path}")
    print("请检查一下 data 文件夹和 csv 文件是不是都在当前路径下！")
else:
    print("正在读取数据，请稍候...")
    df = pd.read_csv(csv_file_path)

    counts_ordered = df['emotion'].value_counts().sort_index()
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    counts_ordered.index = emotion_names

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    counts_ordered.plot(kind='bar', figsize=(10, 5), color='skyblue', edgecolor='black')

    plt.title('FER2013 Emotion Distribution', fontsize=16)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('emotion_distribution.png', dpi=300)
    print("✅ 柱状图已保存为 emotion_distribution.png")
    plt.show()


df = pd.read_csv('data/fer2013.csv')

first_pixel_str = df['pixels'].iloc[0]

pixel_array = np.array(first_pixel_str.split(), dtype='float32')

face_img = pixel_array.reshape(48, 48)

plt.imshow(face_img, cmap='gray')
plt.title(f"Emotion Label: {df['emotion'].iloc[0]}")
plt.axis('off')
plt.show()