from mazeEnv import MazeEnv

env = MazeEnv(render_mode="human")  # Test with human mode first

env.manual_step()

# obs, _ = env.reset()
# rgb_list = []
# rgba_array_list = []
# rgba_buffer_list = []
# while True :
#     a = env.render()
#     action = int(input())
#     obs, r, term, trunc, _ = env.step(action)
#     rgb_list.append(a)
#      # Manually call render to see if it moves
