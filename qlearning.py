import numpy as np
import matplotlib.pyplot as plt

class MazeEnv:
    """迷宫环境"""
    def __init__(self):
        # 5x5的迷宫，0表示空地，1表示墙壁，2表示目标
        self.maze = [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 2]
        ]
        self.rows = 5
        self.cols = 5
        self.reset()
        
    def reset(self):
        """重置环境，回到起点"""
        self.robot_pos = [0, 0]  # 机器人初始位置
        return tuple(self.robot_pos)
    
    def step(self, action):
        """执行动作，返回新状态、奖励和是否结束"""
        # 动作：0-上，1-右，2-下，3-左
        row, col = self.robot_pos
        new_row, new_col = row, col
        done = False
        
        if action == 0:  # 上
            new_row -= 1
        elif action == 1:  # 右
            new_col += 1
        elif action == 2:  # 下
            new_row += 1
        elif action == 3:  # 左
            new_col -= 1
            
        # 检查是否撞墙或越界
        if (new_row < 0 or new_row >= self.rows or 
            new_col < 0 or new_col >= self.cols or 
            self.maze[new_row][new_col] == 1):
            # 撞墙，位置不变，给予惩罚
            reward = -1
        else:
            # 移动成功
            self.robot_pos = [new_row, new_col]
            new_row, new_col = self.robot_pos
            
            # 检查是否到达目标
            if self.maze[new_row][new_col] == 2:
                reward = 10  # 到达目标，给予奖励
                done = True
            else:
                reward = -0.1  # 每步轻微惩罚，鼓励最短路径
                done = False
                
        return tuple(self.robot_pos), reward, done
    
    def render(self):
        """绘制迷宫和机器人位置"""
        for i in range(self.rows):
            for j in range(self.cols):
                if [i, j] == self.robot_pos:
                    print("R", end=" ")  # 机器人
                elif self.maze[i][j] == 1:
                    print("#", end=" ")  # 墙壁
                elif self.maze[i][j] == 2:
                    print("G", end=" ")  # 目标
                else:
                    print(".", end=" ")  # 空地
            print()
        print()


class QLearningRobot:
    """基于Q-Learning的机器人"""
    def __init__(self, env, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate  # 学习率α
        self.gamma = gamma  # 折扣因子γ
        self.epsilon = epsilon  # ε-贪婪策略的ε值
        
        # 初始化Q表，状态是位置(row, col)，动作是0-3
        self.q_table = {}
        for i in range(env.rows):
            for j in range(env.cols):
                self.q_table[(i, j)] = [0.0, 0.0, 0.0, 0.0]  # 上下左右四个动作
        
    def choose_action(self, state):
        """基于ε-贪婪策略选择动作"""
        if np.random.uniform(0, 1) < self.epsilon:
            # 随机选择动作（探索）
            return np.random.choice(4)
        else:
            # 选择Q值最大的动作（利用）
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """更新Q值"""
        # 当前Q值
        current_q = self.q_table[state][action]
        # 新状态的最大Q值
        max_next_q = max(self.q_table[next_state])
        # Q值更新公式
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train(self, episodes=1000):
        """训练机器人"""
        rewards = []  # 记录每回合的总奖励
        steps = []    # 记录每回合的步数
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                step += 1
                
                # 防止无限循环
                if step > 1000:
                    break
            
            rewards.append(total_reward)
            steps.append(step)
            
            # 每100回合打印一次进度
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Steps: {step}")
        
        return rewards, steps
    
    def test(self):
        """测试训练好的机器人"""
        state = self.env.reset()
        self.env.render()
        done = False
        step = 0
        
        while not done and step < 100:
            action = np.argmax(self.q_table[state])  # 只使用利用，不探索
            state, _, done = self.env.step(action)
            self.env.render()
            step += 1


# 主程序
if __name__ == "__main__":
    # 创建环境和机器人
    env = MazeEnv()
    robot = QLearningRobot(env, learning_rate=0.1, gamma=0.9, epsilon=0.1)
    
    # 训练机器人
    print("开始训练...")
    rewards, steps = robot.train(episodes=1000)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("每回合总奖励")
    plt.xlabel("回合数")
    plt.ylabel("总奖励")
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title("每回合步数")
    plt.xlabel("回合数")
    plt.ylabel("步数")
    
    plt.tight_layout()
    plt.show()
    
    # 测试训练好的机器人
    print("测试训练好的机器人:")
    robot.test()

