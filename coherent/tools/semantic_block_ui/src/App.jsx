import { useState, useCallback } from 'react';
import { ReactFlowProvider } from 'reactflow';
import BlockPalette from './components/BlockPalette';
import Canvas from './components/Canvas';
import PropertyEditor from './components/PropertyEditor';
import { useSemanticGraph } from './hooks/useSemanticGraph';
import { exportToJSON, importFromJSON } from './utils/persistence';

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    updateNodeData,
    setNodes,
    setEdges
  } = useSemanticGraph();

  // Find the selected node to pass to PropertyEditor
  const selectedNode = nodes.find((node) => node.selected);

  const handleExport = () => {
    const json = exportToJSON(nodes, edges);
    const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `semantic-block-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        const { nodes: newNodes, edges: newEdges } = importFromJSON(json);
        setNodes(newNodes);
        setEdges(newEdges);
      } catch (err) {
        console.error('Failed to import JSON', err);
        alert('Invalid JSON file');
      }
    };
    reader.readAsText(file);
    // Reset input
    event.target.value = '';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      {/* Top Bar */}
      <header style={{
        height: '48px',
        borderBottom: '1px solid hsl(var(--border-subtle))',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0 1rem',
        backgroundColor: 'hsl(var(--bg-panel))'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ fontWeight: 600, color: 'hsl(var(--accent-primary))' }}>Semantic Block UI</span>
          <span style={{ marginLeft: '1rem', fontSize: '0.8rem', color: 'hsl(var(--text-dim))' }}>MVP v0.1.0</span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <label
            style={{
              padding: '0.4em 0.8em',
              fontSize: '0.8em',
              cursor: 'pointer',
              backgroundColor: 'hsl(var(--bg-element))',
              border: '1px solid hsl(var(--border-subtle))',
              borderRadius: '6px',
              color: 'hsl(var(--text-primary))'
            }}
          >
            Import JSON
            <input type="file" accept=".json" style={{ display: 'none' }} onChange={handleImport} />
          </label>
          <button onClick={handleExport} style={{ padding: '0.4em 0.8em', fontSize: '0.8em' }}>Export JSON</button>
        </div>
      </header>

      {/* Main Workspace */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left Sidebar: Palette */}
        <aside style={{
          width: '240px',
          borderRight: '1px solid hsl(var(--border-subtle))',
          backgroundColor: 'hsla(var(--bg-panel), 0.5)',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ padding: '1rem', borderBottom: '1px solid hsl(var(--border-subtle))' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'hsl(var(--text-secondary))' }}>PALETTE</span>
          </div>
          <div style={{ padding: '1rem' }}>
            <BlockPalette />
          </div>
        </aside>

        {/* Center Canvas */}
        <main style={{ flex: 1, position: 'relative', backgroundColor: 'hsl(var(--bg-app))' }}>
          <ReactFlowProvider>
            <Canvas
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              addNode={addNode}
            />
          </ReactFlowProvider>
        </main>

        {/* Right Sidebar: Properties */}
        <aside style={{
          width: '300px',
          borderLeft: '1px solid hsl(var(--border-subtle))',
          backgroundColor: 'hsl(var(--bg-panel))',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ padding: '1rem', borderBottom: '1px solid hsl(var(--border-subtle))' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'hsl(var(--text-secondary))' }}>PROPERTIES</span>
          </div>
          <div style={{ padding: '1rem' }}>
            <PropertyEditor selectedNode={selectedNode} updateNodeData={updateNodeData} />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
