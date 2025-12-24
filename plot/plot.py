import matplotlib.pyplot as plt
import numpy as np
import os

def plot_instance(vrp, filename):
    """
    Plots the solution.
    - Depots as triangles
    - Customers as circles
    - Routes as lines
    """
    sol = vrp.solution

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot depots as triangles
    ax.scatter(
        vrp.locations[vrp.depot_indices][:, 0],
        vrp.locations[vrp.depot_indices][:, 1],
        marker="^",
        s=150,
        color='red',
        label="Depots"
    )

    # Add depot IDs next to the points
    for depot_id in vrp.depot_indices:
        x, y = vrp.locations[depot_id]
        ax.text(
            x + 0.005, y + 0.005, str(depot_id),
            fontsize=8,
            color='red'
        )

    # Plot customers as circles
    ax.scatter(
        vrp.locations[vrp.customer_indices, 0],
        vrp.locations[vrp.customer_indices, 1],
        marker="o",
        s=50,
        color='blue',
        label="Customers"
    )

    # Add customer IDs next to the points
    for customer_id in vrp.customer_indices:
        x, y = vrp.locations[customer_id]
        ax.text(
            x + 0.005, y + 0.005, str(customer_id),
            fontsize=8,
            color='blue'
        )

    # Plot tours as lines
    # plot only tours of len > 1
    tours = [tour for tour in sol if len(tour) >1]
    for i, tour in enumerate(tours):
        route = [vrp.locations[p[0]] for p in tour]
        route = list(zip(*route))  # unzip into two lists: x, y
        ax.plot(route[0], route[1], alpha=0.6, label=f"Route {i}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("MDVRP Solution")
    ax.legend()
    plt.savefig(f'{filename}')
    plt.close()

