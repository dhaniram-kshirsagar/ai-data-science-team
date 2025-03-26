from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_data_science_team.agents.graph_schema_agent import GraphSchemaAgent
from langchain_openai import ChatOpenAI
import os
import tempfile
import plotly.graph_objects as go
import io
import base64

class UploadResponse(BaseModel):
    message: str
    path: str

class SchemaResponse(BaseModel):
    message: str

class SchemaResult(BaseModel):
    schema: dict
    cypher: str
    graph_image: str  # Base64 encoded image

def generate_graph_image(schema):
    """Generate Plotly graph image from schema"""
    if not schema or not isinstance(schema, dict):
        return None
        
    # Create nodes and edges
    nodes = [node.get('label', f'Node_{i}') for i, node in enumerate(schema.get('nodes', []))]
    edges = [(rel['startNode'], rel['endNode']) 
             for rel in schema.get('relationships', [])]
    
    # Create graph visualization
    edge_trace = go.Scatter(
        x=[], y=[], 
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        text=nodes,
        marker=dict(
            size=20,
            line=dict(width=2)
        )
    )
    
    # Add positions to nodes and edges
    for i, node in enumerate(nodes):
        node_trace['x'] += (i,)
        node_trace['y'] += (i % 2,)
        
    for edge in edges:
        x0, y0 = nodes.index(edge[0]), nodes.index(edge[0]) % 2
        x1, y1 = nodes.index(edge[1]), nodes.index(edge[1]) % 2
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    # Convert to base64 image
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

app = FastAPI(
    title="Graph Schema Generator API",
    description="API for generating Neo4j graph schemas from CSV data",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize LLM
llm = ChatOpenAI(
    model='gpt-4',
    api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0
)

agent_instance = None

@app.post('/upload', response_model=UploadResponse, tags=["File Operations"])
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file for schema generation.
    
    Args:
        file (UploadFile): The CSV file to upload
        
    Returns:
        UploadResponse: Message and temporary file path
    """
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Initialize agent with uploaded file
        global agent_instance
        agent_instance = GraphSchemaAgent(
            model=llm,
            csv_path=temp_path,
            log=True,
            log_path='logs'
        )

        return {'message': 'File uploaded successfully', 'path': temp_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/generate-schema', response_model=SchemaResponse, tags=["Schema Generation"])
async def generate_schema():
    """Generate Neo4j schema from uploaded CSV file.
    
    Returns:
        SchemaResponse: Success message
    """
    if not agent_instance:
        raise HTTPException(status_code=400, detail='No file uploaded')

    try:
        response = agent_instance.invoke_agent()
        return {'message': 'Schema generated successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/get-schema', response_model=SchemaResult, tags=["Schema Retrieval"])
async def get_schema():
    """Retrieve generated schema and Cypher statements.
    
    Returns:
        SchemaResult: Generated schema and Cypher statements
    """
    if not agent_instance:
        raise HTTPException(status_code=400, detail='No file uploaded')

    schema = agent_instance.get_schema()
    cypher = agent_instance.get_cypher() or ""
    graph_image = generate_graph_image(schema) if schema else ""

    # Debug logging
    print(f"Schema: {schema}")
    print(f"Graph image generated: {bool(graph_image)}")
    print(f"Image length: {len(graph_image)}")
    
    if not schema:
        raise HTTPException(status_code=404, detail='Schema not generated')

    return {
        'schema': schema,
        'cypher': cypher,
        'graph_image': graph_image
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
