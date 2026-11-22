import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 让 matplotlib 能正常显示中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FERProcessor:
    def __init__(self,csv_path):
        print("正在加载数据")
        self.df = pd.read_csv(csv_path)

    def split_data(self):
        train_data = self.df[self.df['Usage'] == 'Training']
        val_data = self.df[self.df['Usage'] == 'PublicTest']
        test_data = self.df[self.df['Usage'] == 'PrivateTest']

        print(f"训练集：{len(train_data)}张，验证集：{len(val_data)}张，测试集：{len(test_data)}张")
        return train_data,val_data,test_data

    def process_image(self,pixel_str):
        arr = np.array(pixel_str.split(),dtype='float32')
        arr = arr/255.0
        img_matrix = arr.reshape(48,48)
        return img_matrix

if __name__ == "__main__":
    processor = FERProcessor('data/fer2013.csv')
    train,val,test = processor.split_data()

    # 随机抽取一行（每次运行结果都可能不同）
    sample = train.sample(n=1).iloc[0]

    sample_pixels = sample['pixels']
    img = processor.process_image(sample_pixels)

    print("图片形状",img.shape)
    print("最小像素值：",img.min(),"最大像素值：",img.max())


    emotion_names = ['生气', '厌恶', '恐惧', '开心', '难过', '惊讶', '平静']
    label = emotion_names[int(sample['emotion'])]

    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title(f"表情：{label}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_face.png', dpi=150)
    print("✅ 已保存为 sample_face.png")
    plt.show()

