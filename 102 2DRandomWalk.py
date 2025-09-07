import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 200      # 粒子数
steps = 1000  # フレーム数（= ステップ数）

# 初期位置
x = np.zeros((N, 2), dtype=int)

# 4方向ランダムステップの辞書
step = np.array([[-1,0],[1,0],[0,-1],[0,1]])

fig = plt.figure()
scat = plt.scatter(x[:,0], x[:,1],alpha=0.4)


R = int(np.ceil(2*np.sqrt(steps)))  # 余裕を持たせる
plt.xlim(-R, R)
plt.ylim(-R, R)
plt.title('Random Walk')


def update(frame):
    # 各粒子の一歩
    move = step[np.random.choice(4,N)]
    np.add(x, move, out=x)
    scat.set_offsets(x)

ani = FuncAnimation(fig, update, frames=steps,interval=50)

plt.show()

# 保存したい場合（必要な方だけ有効化）
# ani.save("ink_walk.gif", writer="pillow", fps=30)
# ani.save("ink_walk.mp4", writer="ffmpeg", fps=30)
