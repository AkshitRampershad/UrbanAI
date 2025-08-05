import plotly.graph_objects as go

def plot_layout(layout_json):
    """
    Render the building layout using Plotly from layout JSON structure.
    """
    fig = go.Figure()

    # Draw building footprint
    if 'footprint' in layout_json:
        x, y = zip(*layout_json['footprint'])
        fig.add_trace(go.Scatter(
            x=x, y=y, fill='toself', name='Footprint', line=dict(color='black')
        ))

    # Draw stairs if provided
    if 'stairs' in layout_json:
        for idx, stair in enumerate(layout_json['stairs']):
            x, y = zip(*stair)
            fig.add_trace(go.Scatter(
                x=x, y=y, fill='toself', name=f'Stair {idx+1}', line=dict(color='gray', dash='dot')
            ))

    fig.update_layout(
        title="Proposed Building Layout",
        xaxis=dict(scaleanchor="y", showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True
    )
    return fig
