import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig1 = plt.figure(figsize=(10,6), dpi=80)
ax1 = fig1.add_subplot(111, aspect='equal')

ax1.add_patch(patches.Rectangle(
                (0.1, 0.2),   # (x,y)
                0.1,          # width
                0.1,          # height
                alpha=0.6
                ))
plt.show()