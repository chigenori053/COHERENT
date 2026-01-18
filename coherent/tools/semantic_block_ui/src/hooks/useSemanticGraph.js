import { useCallback } from 'react';
import { useNodesState, useEdgesState, addEdge, MarkerType } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';
import { ROLES, STATUSES } from '../utils/constants';

export const useSemanticGraph = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    const onConnect = useCallback(
        (params) => setEdges((eds) => addEdge({
            ...params,
            type: 'default', // standard semantic edge
            markerEnd: { type: MarkerType.ArrowClosed },
            style: { strokeWidth: 2 }
        }, eds)),
        [setEdges],
    );

    const addNode = useCallback((role, position) => {
        const newNode = {
            id: uuidv4(),
            type: 'semanticBlock',
            position,
            data: {
                role,
                expression: '',
                status: STATUSES.DRAFT,
                created_at: new Date().toISOString()
            },
        };
        setNodes((nds) => nds.concat(newNode));
    }, [setNodes]);

    const updateNodeData = useCallback((id, newData) => {
        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    return { ...node, data: { ...node.data, ...newData } };
                }
                return node;
            })
        );
    }, [setNodes]);

    return {
        nodes,
        edges,
        onNodesChange,
        onEdgesChange,
        onConnect,
        addNode,
        updateNodeData,
        setNodes, // exposed for import
        setEdges, // exposed for import
    };
};
