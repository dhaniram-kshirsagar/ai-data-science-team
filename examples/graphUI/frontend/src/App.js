import React, { useState } from 'react';
import axios from 'axios';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';

function App() {
  const [file, setFile] = useState(null);
  const [schema, setSchema] = useState(null);
  const [cypher, setCypher] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { data } = await axios.get('http://localhost:8000/build-schema');

      setSchema(data.schema);
      setCypher(data.cypher);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const renderSchema = () => {
    if (!schema) return null;

    return (
      <div>
        <h2>Graph Schema</h2>
        <div style={{ display: 'flex', gap: '20px' }}>
          <div style={{ width: '50%' }}>
            <h3>Nodes</h3>
            <ul>
              {Object.entries(schema.nodes).map(([nodeType, properties]) => (
                <li key={nodeType}>
                  <strong>{nodeType}</strong>
                  <ul>
                    {Object.keys(properties).map(prop => (
                      <li key={prop}>{prop}</li>
                    ))}
                  </ul>
                </li>
              ))}
            </ul>
          </div>
          <div style={{ width: '50%' }}>
            <h3>Relationships</h3>
            <ul>
              {schema.relationships.map((rel, index) => (
                <li key={index}>
                  {rel.startNode} → {rel.endNode} [{rel.type}]
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <h1>Graph Schema Generator</h1>
      <div>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? 'Processing...' : 'Upload & Generate Schema'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {schema && (
    <div>
            {console.log('Image data:', schema.graph_image)}

        {renderSchema()}
        <div>
            <h2>Graph Visualization</h2>
            {console.log('Image data:', schema.graph_image)}

            {schema.graph_image && (
                <img 
                    src={`data:image/png;base64,${schema.graph_image.trim()}`} 
                    alt="Graph visualization"
                    style={{ maxWidth: '100%' }}
                    onError={(e) => {
                        console.error('Image failed to load', e);
                        console.log('Image data:', schema.graph_image.slice(0, 100));
                    }}
                />
            )}
        </div>
        <h2>Generated Schema JSON</h2>
        <SyntaxHighlighter language="json" style={docco}>
            {JSON.stringify(schema, null, 2)}
        </SyntaxHighlighter>
    </div>
)}
      {cypher && (
        <div>
          <h2>Generated Cypher</h2>
          <SyntaxHighlighter language="cypher" style={docco}>
            {cypher}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  );
}

export default App;
