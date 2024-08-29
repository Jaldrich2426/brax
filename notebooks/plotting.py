import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable

COLOURS = ["tab:orange",
           "tab:pink",
           "tab:cyan",
           "tab:green",
           "tab:purple",
           "tab:red",
           "tab:blue",
           "tab:olive",
           "tab:brown",
           "tab:grey"]


def draw_belief_traj(ax, states, trajs, map_img, lims,
                     goals=None, vels=None, rollout_fn=None, traces=None, trace_alpha=0.6,
                     robot_radius=0.3, traj_alpha=0.4, collision_radius=None):
    N = states.shape[0]

    # Make colours same length as number of robots, adding duplicates if necessary.
    colours = COLOURS
    while len(colours) < N:
        colours += COLOURS
    colours = colours[:N]

    ax.imshow(map_img, extent=lims, cmap="Greys", zorder=-1)

    if vels is not None:
        xs, ys, dxs, dys = vels  # Magnitudes must be 1.
        mags = np.sqrt(dxs**2 + dys**2)
        # Check for zeros and make the arrows point straight up in that case.
        dys[np.isclose(mags, 0)] = 1
        mags[np.isclose(mags, 0)] = 1

        # Normalize to 1.
        dxs = dxs / mags
        dys = dys / mags

    if not isinstance(rollout_fn, Iterable):
        rollout_fn = [rollout_fn for _ in range(N)]

    for n in range(N):
        if traces is not None:
            # Draw the path so far if provided.
            ax.plot(traces[n, :, 0], traces[n, :, 1], c=colours[n], alpha=trace_alpha, linewidth=3, zorder=0)

        if trajs is not None:
            # print(rollout_fn.shape)
            # print(len(rollout_fn[n](trajs[n], states[n])))
            traj = rollout_fn[n](trajs[n], states[n])[-1].cpu().numpy()

            for i in range(traj.shape[0]):
                ax.plot(traj[i, :, 0], traj[i, :, 1], c=colours[n], alpha=traj_alpha, zorder=0)

        if goals is not None:
            ax.scatter(goals[n, 0], goals[n, 1], c=colours[n], marker="x")

        if vels is not None:
            # Draw the current velocities.
            shift = robot_radius * .66
            ax.arrow(xs[n] - dxs[n] * shift, ys[n] - dys[n] * shift, dxs[n] / 100, dys[n] / 100,
                     color="white", head_width=robot_radius, zorder=2)

        # Draw robot pose.
        ax.add_patch(plt.Circle(states[n, 0, :2], robot_radius, color=colours[n]))

        if collision_radius is not None:
            # Draw tolerance radius.
            ax.add_patch(plt.Circle(states[n, 0, :2], collision_radius, facecolor="none", edgecolor="r"))

    # # Draw the current nodes.
    # ax.scatter(states[..., 0].cpu().numpy(), states[..., 1].cpu().numpy(), c=colours, marker="o", s=400, zorder=1)

    ax.set_xlim(*lims[:2])
    ax.set_ylim(*lims[2:])

def draw_paritcles(ax, particles, map_img, lims,
                     goals=None, vels=None,
                     robot_radius=0.3, traj_alpha=0.4, collision_radius=None):
    num_particles = particles.shape[1]
    horizon = particles.shape[0]

    # Make colours same length as number of robots, adding duplicates if necessary.
    colours = COLOURS
    while len(colours) < horizon:
        colours += COLOURS
    colours = colours[:horizon]

    ax.imshow(map_img, extent=lims, cmap="Greys", zorder=-1)

    if vels is not None:
        xs, ys, dxs, dys = vels  # Magnitudes must be 1.
        xs=xs.cpu().numpy()
        ys=ys.cpu().numpy()
        dxs=dxs.cpu().numpy()
        dys=dys.cpu().numpy()
        mags = np.sqrt(dxs**2 + dys**2)
        # Check for zeros and make the arrows point straight up in that case.
        dys[np.isclose(mags, 0)] = 1
        mags[np.isclose(mags, 0)] = 1

        # Normalize to 1.
        dxs = dxs / mags
        dys = dys / mags

    # if not isinstance(rollout_fn, Iterable):
    #     rollout_fn = [rollout_fn for _ in range(h)]

    # ax.scatter(particles[:, :, 0], particles[:, :, 1], c=colours[h])
    particles = particles.cpu().numpy()
    if goals is not None:
        goals = goals.cpu().numpy()
    for h in range(horizon):
        # if traces is not None:
        #     # Draw the path so far if provided.
        #     ax.plot(traces[n, :, 0], traces[n, :, 1], c=colours[n], alpha=trace_alpha, linewidth=3, zorder=0)

        # if trajs is not None:
            # print(rollout_fn.shape)
            # print(len(rollout_fn[n](trajs[n], states[n])))
        # traj = rollout_fn[n](trajs[n], states[n])[-1].cpu().numpy()

        # for i in range(num_particles):
            # ax.plot(particles[h,i, 0], particles[h,i, 1], c=colours[h], alpha=traj_alpha, zorder=0)
        
        ax.scatter(particles[h,:, 0], particles[h,:, 1], c=colours[h])

        if goals is not None:
            ax.scatter(goals[0, 0], goals[0, 1], c=colours[h], marker="x")

        if vels is not None:
            # Draw the current velocities.
            shift = robot_radius * .66
            # print(xs.shape)
            # print(dxs.shape)
            # print(ys.shape)
            # print(dys.shape)
            ax.arrow(xs[0] - dxs[0] * shift, ys[0] - dys[0] * shift, dxs[0] / 100, dys[0] / 100,
                     color="white", head_width=robot_radius, zorder=2)

        # Draw robot pose.
        if(h==0):
            ax.add_patch(plt.Circle(particles[h, 0, :2], robot_radius, color=colours[h]))

        # if collision_radius is not None:
        #     # Draw tolerance radius.
        #     ax.add_patch(plt.Circle(particles[h, 0, :2], collision_radius, facecolor="none", edgecolor="r"))

    # # Draw the current nodes.
    # ax.scatter(states[..., 0].cpu().numpy(), states[..., 1].cpu().numpy(), c=colours, marker="o", s=400, zorder=1)

    ax.set_xlim(*lims[:2])
    ax.set_ylim(*lims[2:])
