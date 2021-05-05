from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit as st
import streamlit.components.v1 as components
import torch
import plotly.graph_objects as go
from torch.optim import SGD, Adam
from plotly.subplots import make_subplots
from optimal_pytorch.coin_betting.torch import Cocob, Regralizer, Recursive, Scinol2

plt.style.use("seaborn-white")


def plot_coherent(show=False):
    """
    Plot coherent function.
    """
    x = np.linspace(-1.5, 1.5, 250)
    y = np.linspace(-1.5, 1.5, 250)
    minimum = (0, 0)
    X, Y = np.meshgrid(x, y)
    Z = coherent([X, Y])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.set_title(
        r"Coherent function, $f(x) = 3 + \sin(5\theta) + \cos(3\theta) * r^2(5/3-r)$"
    )
    ax.plot(*minimum, "gD")
    if show:
        plt.plot()
        plt.show()
    return fig, ax


def plot_rosenbrock(show=False):
    """
    Plot Rosenbrock function.
    """
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.set_title("Rosenbrock function")
    ax.plot(*minimum, "gD")
    if show:
        plt.plot()
        plt.show()
    return fig, ax


def plot_rastrigin(show=False):
    """
    Plot Rastrigin function.
    """
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap="jet")
    ax.set_title("Rastrigin function")
    ax.plot(*minimum, "gD")
    if show:
        plt.plot()
        plt.show()
    return fig, ax


def coherent(tensor):
    """
    Compute coherent function.
    """
    x, y = tensor
    lib = torch if isinstance(x, torch.Tensor) else np
    r = lib.sqrt(x ** 2 + y ** 2)
    if lib == np:
        theta = np.arctan2(y, x)
    else:
        theta = torch.atan2(y, x)
    return (3 + lib.sin(5 * theta) + lib.cos(3 * theta)) * r ** 2 * (5 / 3 - r)


def rastrigin(tensor):
    """
    Compute Rastrigin function.
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    x, y = tensor
    lib = torch if isinstance(x, torch.Tensor) else np
    A = 10
    f = A * 2 + (x ** 2 - A * lib.cos(x * pi * 2)) + (y ** 2 - A * lib.cos(y * pi * 2))
    return f


def rosenbrock(tensor):
    """
    Compute Rosenbrock function.
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def draw(i, X, Y, pathline, point):
    """
    Draw the optimization path.
    """
    x = X[i]
    y = Y[i]
    pathline[0].set_data(X[: i + 1], Y[: i + 1])
    point[0].set_data(x, y)
    return pathline[0], point[0]


@st.cache(suppress_st_warning=True)
def plotly_rastrigin():
    """
    Plot Rastrigin function with plotly.
    """
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y])
    fig = make_subplots(
        rows=1, cols=1, specs=[[{"is_3d": True}]], subplot_titles=["Rastrigin function"]
    )
    fig.add_trace(go.Surface(x=X, y=Y, z=Z), 1, 1)
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


@st.cache(suppress_st_warning=True)
def plotly_rosenbrock():
    """
    Plot Rosenbrock function with plotly.
    """
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"is_3d": True}]],
        subplot_titles=["Rosenbrock function"],
    )
    fig.add_trace(go.Surface(x=X, y=Y, z=Z), 1, 1)
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


@st.cache(suppress_st_warning=True)
def plotly_coherent():
    """
    Plot coherent function with plotly.
    """
    x = np.linspace(-1.5, 1.5, 250)
    y = np.linspace(-1.5, 1.5, 250)
    X, Y = np.meshgrid(x, y)
    Z = coherent([X, Y])
    fig = make_subplots(
        rows=1, cols=1, specs=[[{"is_3d": True}]], subplot_titles=["Coherent function"]
    )
    fig.add_trace(go.Surface(x=X, y=Y, z=Z), 1, 1)
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        ),
        showlegend=False,
    )
    fig.update_layout(autosize=False, width=800, height=800)
    return fig


def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=500):
    """Run the optimizer."""
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


def frame_selector_ui():
    """Streamlit UI."""
    st.sidebar.markdown("# Parameters")

    fun_select = st.sidebar.radio(
        "Function to optimize:",
        ("Rosenbrock", "Rastrigin", "Coherent"),
    )

    iterations = st.sidebar.slider("iterations:", 500, 1000, step=100)

    # The user can pick which type of object to search for.
    algo = st.sidebar.selectbox(
        "Optimizer?", ["SGD", "Adam", "Cocob", "Regralizer", "Recursive"], 0
    )

    params = {}
    if algo in ["Cocob", "Recursive"]:
        # Choose initial wealth
        # eps = st.sidebar.slider("Choose initial wealth:", 0.1, 10.0, step=0.1)
        # params["eps"] = eps
        if algo == "Recursive":
            momentum = st.sidebar.selectbox("Momentum:", [0, 0.9, 0.99])
            params["momentum"] = momentum
            inner = st.sidebar.selectbox("Inner:", ["Scinol2", "Cocob"])
            params["inner"] = Scinol2 if inner == "Scinol2" else Cocob
    else:
        lr = st.sidebar.slider("Learning rate:", 0.1, 1.0, value=0.1, step=0.1)
        params["lr"] = lr
        if algo == "SGD":
            momentum = st.sidebar.selectbox("Momentum:", [0, 0.9, 0.99])
            params["momentum"] = momentum

    return fun_select, algo, params, iterations


def animate_function(fig, ax, steps, n_frames=100):
    """
    A simple way to produce an animation with Streamlit.
    """
    X = steps[0, :]
    Y = steps[1, :]
    pathline = ax.plot(X[0], Y[0], color="r", lw=1)
    point = ax.plot(X[0], Y[0], "ro")
    point_ani = animation.FuncAnimation(
        fig,
        draw,
        frames=n_frames,
        fargs=(X, Y, pathline, point),
        interval=100,
        blit=True,
        repeat=False,
    )

    # video rendering
    with st.spinner("Wait for it..."):
        with open("myvideo.html", "w") as f:
            print(point_ani.to_html5_video(), file=f)
    st.success("Done!")
    st.markdown("Green point is the minimum:")
    with open("myvideo.html", "r") as HtmlFile:
        source_code = HtmlFile.read()
    components.html(source_code, height=900, width=900)


def run_the_app(function, selected_algo, selected_params, iterations):
    """
    Streamlit core function.
    """

    #  create figure
    if function == "Rosenbrock":
        fig, ax = plot_rosenbrock()
        initial_state = (-2.0, 2.0)
    elif function == "Rastrigin":
        fig, ax = plot_rastrigin()
        initial_state = (-2.0, 3.5)
    elif function == "Coherent":
        fig, ax = plot_coherent()
        initial_state = (0.9, 0.3)
    else:
        st.error("Please select a different function.")
        return

    if selected_algo is None:
        st.error("Please select a different algorithm.")
        return

    optimizers = {
        "Cocob": Cocob,
        "SGD": SGD,
        "Adam": Adam,
        "Regralizer": Regralizer,
        "Recursive": Recursive
    }
    functions = {
        "Rosenbrock": rosenbrock,
        "Rastrigin": rastrigin,
        "Coherent": coherent,
    }

    # run the algorithm
    optimizer = optimizers[selected_algo]
    function = functions[function]
    tot_iter = iterations
    steps = execute_steps(
        function, initial_state, optimizer, selected_params, num_iter=tot_iter
    )

    # function animation
    animate_function(fig, ax, steps)


if __name__ == "__main__":
    st.title("Coin Betting algorithms visualized")

    # Add a selector for the app mode on the sidebar.
    st.sidebar.title("Command line:")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Run optimizers"]
    )
    if app_mode == "Show instructions":
        description = st.markdown(
            "In this app we compare coin-betting optimizers with SGD and Adam on non-convex \
            functions notoriously difficult to optimize.\n\
            To visualize the possible functions please use the buttons on the left."
        )
        st.sidebar.success('To continue select "Run optimizers".')

        images = {
            "Rastrigin": plotly_rastrigin(),
            "Rosenbrock": plotly_rosenbrock(),
            "Coherent": plotly_coherent(),
        }

        fun = st.sidebar.radio(
            "Function:",
            ("Rosenbrock", "Rastrigin", "Coherent"),
        )

        im = st.empty()
        figure = images[fun]
        im.write(figure)

    elif app_mode == "Run optimizers":
        st.write("Choose the options on the left.")
        (
            loss_function,
            algorithm,
            hyperparams,
            time_horizon,
        ) = frame_selector_ui()
        if st.sidebar.button("Run!"):
            run_the_app(loss_function, algorithm, hyperparams, time_horizon)
