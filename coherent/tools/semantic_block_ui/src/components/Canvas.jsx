import React, { useCallback, useMemo, useRef } from 'react';
import ReactFlow, {
    Background,
    Controls,
    MiniMap,
    useReactFlow
} from 'reactflow';
import 'reactflow/dist/style.css';
import SemanticBlock from './nodes/SemanticBlock';

const Canvas = ({
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode
}) => {
    const reactFlowWrapper = useRef(null);
    const { screenToFlowPosition } = useReactFlow();

    const nodeTypes = useMemo(() => ({ semanticBlock: SemanticBlock }), []);

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();

            const role = event.dataTransfer.getData('application/reactflow');

            // check if the dropped element is valid
            if (typeof role === 'undefined' || !role) {
                return;
            }

            // precise positioning using ReactFlow hook
            const position = screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });

            addNode(role, position);
        },
        [addNode, screenToFlowPosition]
    );

    return (
        <div
            style={{ width: '100%', height: '100%' }}
            ref={reactFlowWrapper}
            onDrop={onDrop}
            onDragOver={onDragOver}
        >
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                fitView
            >
                <Background color="#444" gap={16} />
                <Controls />
                <MiniMap
                    nodeColor={(n) => '#555'}
                    maskColor="rgba(0, 0, 0, 0.2)"
                    style={{ backgroundColor: '#222' }}
                />
            </ReactFlow>
        </div>
    );
};

export default Canvas;
