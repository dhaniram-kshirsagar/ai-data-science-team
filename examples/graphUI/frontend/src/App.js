import React, { useState } from 'react';
import axios from 'axios';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';

function App() {
  const [filePath, setFilePath] = useState('');
  const [schema, setSchema] = useState(null);
  const [cypher, setCypher] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [saveMessage, setSaveMessage] = useState(null);
  const [savePath, setSavePath] = useState('');

  const handleFilePathChange = (e) => {
    setFilePath(e.target.value);
  };

  const handleSavePathChange = (e) => {
    setSavePath(e.target.value);
  };

  const handleGenerateFromPath = async () => {
    if (!filePath) return;

    setLoading(true);
    setError(null);
    setSaveMessage(null);

    try {
      const { data } = await axios.post('http://localhost:8000/build-schema', {
        file_path: filePath
      });

      setSchema(data.schema);
      setCypher(data.cypher);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSchema = async () => {
    if (!schema) return;

    setLoading(true);
    setError(null);
    setSaveMessage(null);

    try {
      const { data } = await axios.post('http://localhost:8000/save-schema', {
        schema: schema,
        output_path: savePath || null
      });

      setSaveMessage(data.message);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to save schema');
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
      
      <div style={{ marginBottom: '20px' }}>
        <h3>Provide a File Path</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input 
            type="text" 
            value={filePath} 
            onChange={handleFilePathChange} 
            placeholder="Enter absolute path to CSV file"
            style={{ minWidth: '300px' }}
          />
          <button onClick={handleGenerateFromPath} disabled={!filePath || loading}>
            {loading ? 'Processing...' : 'Generate Schema from Path'}
          </button>
        </div>
      </div>

      {error && <div className="error" style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
      {saveMessage && <div className="success" style={{ color: 'green', marginTop: '10px' }}>{saveMessage}</div>}
      
      {schema && (
        <div>
          <div style={{ marginBottom: '20px', marginTop: '20px' }}>
            <h3>Save Schema to File</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input 
                type="text" 
                value={savePath} 
                onChange={handleSavePathChange} 
                placeholder="Enter save path (optional)"
                style={{ minWidth: '300px' }}
              />
              <button onClick={handleSaveSchema} disabled={loading}>
                {loading ? 'Saving...' : 'Save Schema'}
              </button>
            </div>
            <div style={{ fontSize: '0.8rem', marginTop: '5px', color: '#666' }}>
              If no path is provided, the schema will be saved in the server's default location.
            </div>
          </div>

          {renderSchema()}
          <div>
            <h2>Graph Visualization</h2>
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
