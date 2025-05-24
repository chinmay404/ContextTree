from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


def draw_graph(graph, name="graph_image"):
    graph_image = graph.get_graph(xray=True).draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER)
    with open(f"{name}.png", "wb") as f:
        f.write(graph_image)
    display(Image(f"{name}.png"))
